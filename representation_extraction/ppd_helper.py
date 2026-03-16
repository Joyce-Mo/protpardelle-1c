import torch
import torch.nn.functional as F
import itertools
import os
from pathlib import Path
from boltz_diffusion import weighted_rigid_align

from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose, initialize_config_dir

from protpardelle.core.models import Protpardelle, load_model
from protpardelle.core.diffusion import noise_coords

from protpardelle.data.atom import atom37_coords_from_bb, atom37_mask_from_aatype, atom73_mask_from_aatype
from protpardelle.data.dataset import make_fixed_size_1d

from protpardelle.data.pdb_io import load_feats_from_pdb, write_coords_to_pdb
from protpardelle.data.sequence import seq_to_aatype
from protpardelle.common import residue_constants

from protpardelle.env import (
    FOLDSEEK_BIN,
    PACKAGE_ROOT_DIR,
    PROTEINMPNN_WEIGHTS,
    PROTPARDELLE_MODEL_PARAMS,
    PROTPARDELLE_OUTPUT_DIR,
)
from protpardelle.utils import (
    apply_dotdict_recursively,
    get_default_device,
    seed_everything,
    namespace_to_dict,
    unsqueeze_trailing_dims
)

"""
TODO
- rename seq_final to something that makes sense or store it in string class
- standardize handling of backbone lenfths in init functions
- better interface with chroma
"""

class ProtpardelleHelper:
    def __init__(
        self,
        max_steps=500,
        device = "cuda",
        model_name = 'cc89',
        epoch = '415',
        seed=None,
    ):
        self.device = device

        checkpoint_path = str(PROTPARDELLE_MODEL_PARAMS / "weights" / f"{model_name}_epoch{epoch}.pth")
        config_path = str(PROTPARDELLE_MODEL_PARAMS / "configs" / f"{model_name}.yaml")
        self.model = load_model(config_path, checkpoint_path)

        #self.noise_schedule = self.model.make_sampling_noise_schedule()
        self.noise_schedule = self.model.sampling_noise_schedule_default
        
        self.num_atoms_per_residue = self.model.n_atoms
        
        self.seq_init = None
        self.seq_final = None
        self.max_t = 1.0
        self.max_steps = max_steps

    def set_mask(self, length):
        """
        Omits certain DoFs from distance computations but still allows them to be updated by
        operations on the string (eg reparameterization)
            Note that True means an atom is included in distance, whereas False is not.

        Returns a mask with shape [1, num residues x num atoms, 3]
        """
        #mask = torch.zeros(size=(1, length, self.num_atoms_per_residue, 3))

        mask37 = atom37_mask_from_aatype(self.seq_final, self.seq_mask)[0:1]
        mask37 = mask37.unsqueeze(dim=-1).expand(-1,-1,-1,3)

        return mask37.reshape(1,-1,3).to(self.device)
    
    def initialize_noise(self, num_backbones, len_backbones):
        """
        Initializes random noise protein structures.
        Inputs:
            - num_backbones: int
            - len_backbones: int
        """
        len_backbones = len_backbones[0]
        print('random init')
        coords = torch.randn(*(num_backbones, len_backbones, self.num_atoms_per_residue, 3)) # may want to clip
        coords *= self.noise_schedule(1.0)

        if self.seq_init is None:
            self.seq_init = torch.zeros((num_backbones, len_backbones, 21))
            self.seq_init[:,:,-1] = 1
            self.seq_init = self.seq_init.int()
            self.seq_final = self.seq_init.clone().to(self.device)

        self.seq_mask = torch.ones(num_backbones, len_backbones).to(self.device)

        return coords.reshape(num_backbones,-1,3).to(self.device)

    def initialize_partial_noise(self, protein_init, len_backbones, t, use_seq=True):
        """
        Initializes a partially noised protein structure given a path to a pdb file.
        """
        assert t <= self.max_t

        if type(protein_init) == str:
            struct_init, seq_init = self.initialize_from_file(protein_init, use_seq=use_seq)
        else:
            struct_init = protein_init
            seq_init = torch.full((1,len_backbones), 21)

        if self.seq_init is None:
            self.seq_init = seq_init.to(torch.int).to(self.device)
            self.seq_final = self.seq_init.clone().to(self.device)

        struct_init = struct_init.to(self.device)

        # expects coords with shape (B, N, A, X)
        #mask = atom37_mask_from_aatype(aatype=self.seq_init[0].argmax(dim=-1))
        mask37 = atom37_mask_from_aatype(self.seq_final)

        struct_init = noise_coords(struct_init.reshape(1,-1,self.num_atoms_per_residue, 3),
                                   self.noise_schedule(torch.tensor(t)).to(self.device),
                                   atom_mask=mask37)
        self.seq_mask = torch.ones(1, len_backbones).to(self.device)

        return struct_init.reshape(1, -1, 3).to(self.device)
    
    def initialize_from_file(self, file_path, use_seq=True):
        """
        Initializes a protein structure given a path to a pdb file.
        """
        feats = load_feats_from_pdb(pdb=file_path,
                                    load_atom73=False,
                                    chain_id=None,
                                    atom14=False,
                                    include_pos_feats=False,)[0]
        
        coords = feats["atom_positions"].to(self.device) # [num_res, num_atom_type, 3]

        if use_seq:
            #self.seq_init = F.one_hot(feats["aatype"], 21).to(self.device)
            self.seq_init = feats["aatype"].to(torch.int).to(self.device) # this will not be 1hot, will have shape [Nres]
        else:
            self.seq_init = torch.zeros((coords.shape[0], 21))
            self.seq_init[:,-1] = 0
        self.seq_final = self.seq_init.clone().to(self.device)
        self.seq_mask = torch.ones(1, coords.shape[0]).to(self.device)

        return coords.reshape(-1,3).unsqueeze(dim=0)
    
    def tensor2atom37coords(self, images_tensor):
        """
        Takes a tensor of size (B, N, 3) and converts to an protpardelle atom37 tensor
        """
        return images_tensor.reshape(images_tensor.shape[0], -1, self.num_atoms_per_residue, 3).to(self.device)
    
    def atom37coords2tensor(self, proteins):
        """
        Takes a protpardelle atom37 tensor and converts to a tensor of size (B, N, 3)
        """
        return proteins.reshape(proteins.shape[0],-1,3).to(self.device)
    
    def save_PDBs(self, proteins, path, basename, itr, prefix='', suffix='', nan=None):
        """
        """
        if basename != '':
            basename += '_'
        proteins = self.tensor2atom37coords(proteins) # protpardelle code expects shape [batch, num_res, num_atom_type, 3]
        
        for idx in range(proteins.shape[0]):
            filepath = os.path.join(path, f'{prefix}{basename}i{itr}_p{idx}{suffix}.pdb')
            seq = self.seq_final[idx]

            write_coords_to_pdb(coords_in=[proteins[idx]], filename=[filepath], aatype=[seq]) # this should be batch-able
            if (proteins[idx].abs() > 999).any().item():
                print(f'Warning: some coords in {filepath} exceed the PDB format column width')

    def stepper(self, images_tensor, tspan, predict_seq=False, seq_mask=None, prev_pred=None, seq_self_cond=None, 
                step_scale=1.0, return_xhat=True, align=True, **kwargs):
        """
        Inputs:
            - images_tensor: a (B, Nx37, 3) tensor of protein structures
            - prev_pred: a (B, N, 37, 3) tensor of protein structures. note that protpardelle rescales this, so
                just using the previous x_hat won't work.
            - seq_self_cond: not used for cc89
            - return_xhat: whether to return perceived denoising targets
            - align: whether to Kabsch align x_t coords to x_hat before taking an Euler step. I've found that
                this quantitatively improves RMSD to ground truth for cc89.
        """
        torch.set_grad_enabled(False)

        # somehow numerical precision gets mixed up, so just set everything to fp32
        dtype = torch.float32

        # st_traj is current discrete sequence, s0_traj is log probs, not relevant for cc89 tho
        xt = self.tensor2atom37coords(images_tensor).to(dtype)
        B, Nres, _, _ = xt.shape

        if self.seq_final.shape[0] != B:
            self.seq_final = self.seq_final[0:1].expand(B,-1) # should have shape B, Nres

        if seq_mask is None:
            if self.seq_mask is None:
                seq_mask = torch.ones(B, Nres).to(self.device)
            else:
                seq_mask = self.seq_mask

        # we absolutely must set the ghost atom xyz's to 0 before denoising
        mask37 = atom37_mask_from_aatype(self.seq_final, seq_mask).bool()
        xt[~mask37] = 0

        if True:
            sigma_curr = self.noise_schedule(torch.tensor(tspan[0]),).expand(B).to(self.device)
            sigma_next = self.noise_schedule(torch.tensor(tspan[1]),).expand(B).to(self.device)
        else:
            sigma_curr = chroma_noise_schedule.sigma(tspan[0]).expand(B).to(self.device)
            sigma_next = chroma_noise_schedule.sigma(tspan[1]).expand(B).to(self.device)

        if type(prev_pred) is torch.Tensor:
            prev_pred = prev_pred.to(dtype)
            #prev_pred[~mask37] = 0 # masking ghost atoms in self-conditioning channel somehow results in worse performance
            # maybe i just need to center this first? idk

        denoised_coords, aatype_logprobs, struct_self_cond_out, _ = self.model.forward(
            noisy_coords=xt,
            noise_level=sigma_curr.to(dtype),
            seq_mask=seq_mask,
            residue_index=torch.arange(1,Nres+1).unsqueeze(dim=0).expand((B,-1)).to(self.device), # starts at 1, shape B N
            chain_index=torch.zeros((B, Nres)).to(self.device),
            struct_self_cond=prev_pred,
            seq_self_cond=seq_self_cond,  # logprobs
            run_struct_model=True,
            run_mpnn_model=predict_seq,
           # adj_cond=torch.zeros((B, N, N)).to(dtype).to(self.device),
        )

        def ode_step(
            sigma_in,
            sigma_next,
            xt_in,
            x0_pred,
            score=None,
            stage2=False,
        ):
            mask = (sigma_in > 0).float()
            score = (xt_in - x0_pred) / unsqueeze_trailing_dims(
                sigma_in.clamp(min=1e-6), xt_in
            )
            score = score * unsqueeze_trailing_dims(mask, score)
            
            step = (score
                    * step_scale
                    * unsqueeze_trailing_dims(sigma_next - sigma_in, score))
            new_xt = xt_in + step
            return new_xt

        if align:
            #denoised_coords = weighted_rigid_align(denoised_coords.reshape(B,-1,3), xt.reshape(B,-1,3), weights=None).reshape(B,Nres,-1,3)
            #struct_self_cond_out = denoised_coords / self.model.sigma_data
            # align xt coords onto xhat, since xhat is used for self-conditioning and we don't want to mess with it
            xt = weighted_rigid_align(xt.reshape(B,-1,3), denoised_coords.reshape(B,-1,3), mask=self.set_mask(Nres)[:,:,0])
            xt = xt.reshape(B,Nres,-1,3)
        
        xt = ode_step(sigma_curr, sigma_next, xt, denoised_coords)

        torch.cuda.empty_cache()

        new_images_tensor = self.atom37coords2tensor(xt)
        xhat_tensor = self.atom37coords2tensor(denoised_coords)
        self.seq_mask = seq_mask
        self.struct_self_cond_out = struct_self_cond_out

        if return_xhat:
            return new_images_tensor, xhat_tensor
        else:
            return new_images_tensor