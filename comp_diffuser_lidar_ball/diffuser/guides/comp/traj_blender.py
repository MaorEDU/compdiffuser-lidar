import numpy as np
import diffuser.utils as utils
from diffuser.models.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
from diffuser.datasets.normalization import DatasetNormalizer

class Traj_Blender:
    def __init__(self, diffusion: Stgl_Sml_GauDiffusion_InvDyn_V1, 
                        normalizer: DatasetNormalizer,
                        blend_type: str,
                        exp_beta=3,
                        ):
        self.exp_beta = exp_beta
        self.diffusion = diffusion
        self.blend_type = blend_type
        ## optional (B,T,D) -> (B,T,D) snap onto realizable observations, injected
        ## by the caller for blend_type='switch_map' (mirrors policy.grid_reranker).
        ## Left None => 'switch_map' degrades to plain 'switch'.
        self.projector = None
        self.len_ovlp = self.diffusion.len_ovlp_cd
        self.hzn_step_size = self.diffusion.horizon - self.len_ovlp
        self.hzn = self.diffusion.horizon
        self.gap_len = self.hzn - 2 * self.len_ovlp
        self.normalizer = normalizer
        assert self.gap_len > 0

    def blend_traj_lists(self, trajs_list, do_unnorm):
        """
        trajs_list: list of len n_comp: [ (B,H,D),..., ]
        returns:
            - trajs_out: a list of [tot_hzn, dim]
        """
        ## to np and unnorm
        trajs_list = utils.get_np_trajs_list(trajs_list, do_unnorm=do_unnorm, 
                                             normalizer=self.normalizer)

        n_comp = len(trajs_list)
        b_s,_, dd = trajs_list[0].shape ## b h d

        tot_hzn = n_comp * self.diffusion.horizon - \
                    (n_comp - 1) * self.diffusion.len_ovlp_cd
        
        print(f'{tot_hzn=}')
        # trajs_out = np.zeros( shape=(b_s, tot_hzn, dd) ) ## NOTE: default is float64
        trajs_out = np.zeros( shape=(b_s, tot_hzn, dd), dtype=np.float32 ) ## Dec 2: changed to float32
        cnt_v = np.zeros_like(trajs_out)
        ## copy non-ovlp parts
        for i_c in range(n_comp):
            tjs_p_i = trajs_list[i_c]

            if i_c == 0:
                tmp_idx_1 = 0
                tmp_idx_2 = self.hzn_step_size
                ## B,hstep,dim
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, :self.hzn_step_size, :]
            elif i_c < n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.gap_len
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:self.len_ovlp+self.gap_len, :]
                
            elif i_c == n_comp - 1:
                tmp_idx_1 = self.hzn + (i_c - 1) * self.hzn_step_size
                tmp_idx_2 = tmp_idx_1 + self.hzn_step_size

                assert tmp_idx_2 == tot_hzn
                trajs_out[:, tmp_idx_1:tmp_idx_2, :] = tjs_p_i[:, self.len_ovlp:, :]

            cnt_v[ :, tmp_idx_1:tmp_idx_2, : ] += 1
            utils.print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}, {tot_hzn=}')

        ## handle and merge the ovlp parts
        for i_c in range(n_comp-1):
            tmp_idx_1 = (i_c + 1) * self.hzn_step_size
            tmp_idx_2 = tmp_idx_1 + self.len_ovlp

            ## b,sm_hzn,d
            tjs_p_i = trajs_list[i_c]
            _, end_tjs_i = self.diffusion.extract_ovlp_from_full(tjs_p_i)
            ## b,sm_hzn,d
            tjs_p_i_plus_1 = trajs_list[i_c+1]
            st_tjs_i_plus_1, _ = self.diffusion.extract_ovlp_from_full(tjs_p_i_plus_1)


            ## b,len_o,d
            trajs_blend = blend_2_np_trajs_23d(end_tjs_i, st_tjs_i_plus_1,
                                               self.blend_type, self.exp_beta,
                                               projector=self.projector)

            trajs_out[:, tmp_idx_1:tmp_idx_2, :] = trajs_blend
            cnt_v[:, tmp_idx_1:tmp_idx_2, :] += 1

            utils.print_color(f'{i_c=} {tmp_idx_1=}, {tmp_idx_2=}')
        assert tmp_idx_2 == (tot_hzn - self.hzn_step_size)


        assert (cnt_v == 1).all()

        

        return trajs_out





def blend_2_np_trajs_23d(traj_1: np.ndarray, traj_2: np.ndarray, blend_type='exponential', beta=5,
                         projector=None):
    """
    ** Only takes in the ovlp parts **, blend full traj_1 and traj_2
    ** Blend for multiple dim,
    ----[----
         ----]-----

    Parameters:
    - traj_1: np.ndarray, shape (N1, D), first trajectory positions
    - traj_2: np.ndarray, shape (N2, D), second trajectory positions
    - blend_type: str, type of blending function ('exponential', 'cosine', 'linear',
      'smoothstep', 'switch', 'switch_map')
    - beta: float, parameter for the exponential blending function (controls sharpness)
    - projector: optional callable (B, T, D) -> (B, T, D) snapping frames onto the set of
      physically realizable observations. Only used by 'switch_map'.

    Returns:
    - traj_blend: np.ndarray, blended trajectory positions

    ---------------------------------------------------------------------------
    2026-07-29: WHY 'switch' EXISTS -- convex averaging is INVALID for LiDAR.
    ---------------------------------------------------------------------------
    Every weighted mode below computes ``w * traj_1 + (1-w) * traj_2`` PER
    CHANNEL. For the o2d model (D=2, xy) the average of two positions is a
    position, so this is harmless -- which is exactly why the o2d control always
    passed. For the LiDAR model (D=30 beam ranges) the average of two scans taken
    from two DIFFERENT poses is in general **not the scan of any pose**: it is an
    unrealizable observation, and the grid localizer cannot place it.

    Measured on the composed mr12 @1.6M dump (ball_plan_anatomy.py), distance to
    the nearest realizable scan in ``grid.free_scans``, by seam phase (overlap =
    phase 0-55 of each 104-frame stride):

        phase   0-8    0.288 wu/beam     (w ~ 1: nearly pure traj_1)
        phase   8-16   0.711             <- peak mixing
        phase  16-24   0.738
        phase  24-32   0.528
        phase  32-40   0.394
        phase  40-56   0.25-0.31
        phase  56-104  0.177-0.241       (segment interior, NO blending)

      overlap-phase 0.423 vs interior-phase 0.221  ->  1.91x worse
      (n_comp=1 bridge plans, no blending at all:   0.079)

    The per-frame MAP decode and the sequence-filtered decode agree to 0.5 wu
    through frame 119 and then diverge permanently to ~23 wu starting at frame
    120 -- inside the first overlap's peak-mixing window (first seam at 104).
    That is the mechanism behind "G2b 22%, end err 35.5 wu": the belief filter is
    derailed by unrealizable blended scans at the first seam and never recovers.
    NOTE this is NOT the old "seam teleport" hypothesis (jump POSITIONS are
    uniform mod 104, rejected 3x) -- the damage is loss of scan realizability,
    which was never measured.

    'switch' takes a hard step at the overlap midpoint, so every output frame is
    a scan some segment actually generated. Candidates are already ranked by
    overlap agreement (``utils.compute_ovlp_dist``), so the two halves are
    chosen to agree there. 'switch_map' additionally snaps to the map manifold.
    """

    # assert traj_1.ndim == 2 and traj_1.shape[1] == 1
    assert traj_1.ndim in [2,3] and traj_2.ndim in [2,3] 
    assert traj_1.shape and traj_2.shape

    if traj_1.ndim == 2:
        len_tj, _ = traj_1.shape
    else:
        b_s, len_tj, _ = traj_1.shape
    # Overlapping region from t = 8 to t = 10
    t_overlap_start = 0
    t_overlap_end = len_tj - 1
    t_overlap = np.arange(0, len_tj) ## 1D

    ## Blending function selection, Checked, correct formula
    if blend_type in ['exponential', 'exp']:
        # Exponential blending function
        def w(t):
            exponent = -beta * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
    elif blend_type == 'cosine':
        # Cosine blending function
        def w(t):
            return 0.5 * (1 + np.cos(np.pi * (t - t_overlap_start) / (t_overlap_end - t_overlap_start)))
    elif blend_type == 'linear':
        # Linear blending function
        def w(t):
            return 1 - (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
    elif blend_type == 'smoothstep':
        # Smoothstep blending function
        def w(t):
            x = (t - t_overlap_start) / (t_overlap_end - t_overlap_start)
            return 1 - (3 * x**2 - 2 * x**3)
    elif blend_type in ['switch', 'switch_map']:
        ## Hard step at the overlap midpoint: NO averaging, so every frame stays a
        ## realizable observation. See the docstring for why averaging is invalid
        ## for a range-vector representation.
        def w(t):
            mid = 0.5 * (t_overlap_start + t_overlap_end)
            return (np.asarray(t) <= mid).astype(np.float64)
    else:
        raise ValueError("Invalid blending function. Choose 'exponential', 'cosine', "
                         "'linear', 'smoothstep', 'switch' or 'switch_map'.")

    ## Compute weights, np 1d, from 0 to 1
    ## weights = w(t_overlap)[:, np.newaxis]  # Column vector for broadcasting
    weights = w(t_overlap)  # Column vector for broadcasting
    if traj_1.ndim == 2:
        weights = weights[:, None] ## (len_tj, 1)
    elif traj_1.ndim == 3:
        weights = weights[None, :, None,] ## (1, len_tj, 1)
    
    # print(f'{weights[(0,-1),]=},{weights.shape=}')

    ## Blend the overlapping region
    traj_blend = weights * traj_1 + (1 - weights) * traj_2

    # print(f'{traj_blend.shape=}')

    if blend_type == 'switch_map' and projector is not None:
        traj_blend = projector(traj_blend)

    return traj_blend
