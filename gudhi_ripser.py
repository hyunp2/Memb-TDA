import numpy               as np
import torch          
from gudhi.rips_complex     import RipsComplex

############################
# Vietoris-Rips filtration #
############################

# The parameters of the model are the point coordinates.

def _Rips(DX, max_edge, dimensions, homology_coeff_field):
    # Parameters: DX (distance matrix), 
    #             max_edge (maximum edge length for Rips filtration), 
    #             dimensions (homology dimensions)

    # Compute the persistence pairs with Gudhi
    rc = RipsComplex(distance_matrix=DX, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=max(dimensions)+1)
    st.compute_persistence(homology_coeff_field=homology_coeff_field)
    pairs = st.flag_persistence_generators()

    L_indices = []
    for dimension in dimensions:

        if dimension == 0:
            finite_pairs = pairs[0]
            essential_pairs = pairs[2]
        else:
            finite_pairs = pairs[1][dimension-1] if len(pairs[1]) >= dimension else np.empty(shape=[0,4])
            essential_pairs = pairs[3][dimension-1] if len(pairs[3]) >= dimension else np.empty(shape=[0,2])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))
    return L_indices     
        
class RipsLayer(torch.nn.Module):
    """
    TensorFlow layer for computing Rips persistence out of a point cloud
    """
    def __init__(self, homology_dimensions, maximum_edge_length=np.inf, min_persistence=None, homology_coeff_field=11, **kwargs):
        """
        Constructor for the RipsLayer class

        Parameters:
            maximum_edge_length (float): maximum edge length for the Rips complex 
            homology_dimensions (List[int]): list of homology dimensions
            min_persistence (List[float]): minimum distance-to-diagonal of the points in the output persistence diagrams (default None, in which case 0. is used for all dimensions)
            homology_coeff_field (int): homology field coefficient. Must be a prime number. Default value is 11. Max is 46337.
        """
        super().__init__(**kwargs)
        self.max_edge = maximum_edge_length
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence != None else [0. for _ in range(len(self.dimensions))]
        self.hcf = homology_coeff_field
        assert len(self.min_persistence) == len(self.dimensions)

        
    def forward(self, X):
        """
        Compute Rips persistence diagram associated to a point cloud

        Parameters:   
            X (TensorFlow variable): point cloud of shape [number of points, number of dimensions]

        Returns:
            List[Tuple[tf.Tensor,tf.Tensor]]: List of Rips persistence diagrams. The length of this list is the same than that of dimensions, i.e., there is one persistence diagram per homology dimension provided in the input list dimensions. Moreover, the finite and essential parts of the persistence diagrams are provided separately: each element of this list is a tuple of size two that contains the finite and essential parts of the corresponding persistence diagram, of shapes [num_finite_points, 2] and [num_essential_points, 1] respectively
        """    
        # Compute distance matrix
        DX = torch.norm(torch.unsqueeze(X, 1)-torch.unsqueeze(X, 0), dim=2)
        # Compute vertices associated to positive and negative simplices 
        # Don't compute gradient for this operation
        indices = _Rips(DX.detach().cpu().numpy(), self.max_edge, self.dimensions, self.hcf)
#         indices = torch.from_numpy(indices).to(DX.device)
        # Get persistence diagrams by simply picking the corresponding entries in the distance matrix
        self.dgms = []
        
        def gather_nd(params, indices):
            return params[indices.tolist()]
            
        for idx_dim, dimension in enumerate(self.dimensions):
            cur_idx = indices[idx_dim]
            cur_idx = list(map(lambda inp: torch.from_numpy(inp).to(X.device), cur_idx ))
            
            if dimension > 0:
                finite_dgm = torch.reshape(gather_nd(DX, torch.reshape(cur_idx[0], [-1,2])), [-1,2])
                essential_dgm = torch.reshape(gather_nd(DX, torch.reshape(cur_idx[1], [-1,2])), [-1,1])
            else:
                reshaped_cur_idx = torch.reshape(cur_idx[0], [-1,3])
                finite_dgm = torch.cat([X.new_zeros([reshaped_cur_idx.shape[0],1]), torch.reshape(gather_nd(DX, reshaped_cur_idx[:,1:]), [-1,1])], dim=1)
                essential_dgm = X.new_zeros([cur_idx[1].shape[0],1])
            min_pers = self.min_persistence[idx_dim]
            if min_pers >= 0:
                persistent_indices = torch.where(torch.abs(finite_dgm[:,1]-finite_dgm[:,0]) > min_pers)
                self.dgms.append((torch.reshape(torch.select(finite_dgm, dim=0, index=persistent_indices),[-1,2]), essential_dgm))
            else:
                self.dgms.append((finite_dgm, essential_dgm))
        return self.dgms

  
if __name__ == "__main__":
    layer = RipsLayer(maximum_edge_length=2., homology_dimensions=[0]).cuda()
    x = torch.randn(50,3).cuda()
    print(layer(x))
