import torch
import numpy as np

"""
BTSP (Binding Through Synchronization and Plasticity) model class.

Attributes:
    input_size (int): Size of the input.
    mem_size (int): Size of the memory.
    fq (float): Possibility hyperparameter for CA1_WTA.
    fp (float): Possibility hyperparameter for CA3_WTA.
    fw (float): Possibility hyperparameter for model learnable weights.
    W_feed (torch.Tensor): Learnable weight matrix for feedforward connections.
    W_back (torch.Tensor): Learnable weight matrix for feedback connections.
    W_feed_mask (torch.Tensor): Binary mask for W_feed.
    W_back_mask (torch.Tensor): Binary mask for W_back.
    CA1_WTA (bool): Flag indicating whether to use Winner-Take-All mechanism in CA1.
    CA3_WTA (bool): Flag indicating whether to use Winner-Take-All mechanism in CA3.
    parallel_btsp (bool): Flag indicating whether to use parallel BTSP.
    thr_ca1 (float): Threshold for CA1 firing.
    thr_ca3 (float): Threshold for CA3 firing.
"""
class BtspModel:

    def __init__(self, input_size, mem_size, fq, fp, fw, device=None):
        """
        Initializes the BTSP model.

        Args:
            input_size (int): Size of the input.
            mem_size (int): Size of the memory.
            fq (float): Possibility hyperparameter for CA1_WTA.
            fp (float): Possibility hyperparameter for CA3_WTA.
            fw (float): Possibility hyperparameter for model learnable weights.
            device (str, optional): Device to run the model on. Defaults to None.
        """

        # model hyperparameters
        self.input_size = input_size
        self.mem_size = mem_size

        # possibility hyperparameter
        self.fq = fq
        self.fp = fp
        self.fw = fw

        # model learnable weights
        self.W_feed = self.W_back = 0
        self.W_feed_mask = (torch.rand(input_size, mem_size) <= fw).bool().to(device)
        self.W_back_mask = (torch.rand(input_size, mem_size) <= fw).bool().to(device)
        

        self.CA1_WTA = False
        self.CA3_WTA = False
        self.parallel_btsp = True

        self.thr_ca1 = None
        self.thr_ca3 = None


    def __topk_thresh__(self, y_sum, top_k):
        """
        Applies top-k thresholding to the input tensor.

        Args:
            y_sum (torch.Tensor): Input tensor.
            top_k (int): Number of top values to keep.

        Returns:
            torch.Tensor: Output tensor after top-k thresholding.
        """

        if not torch.any(y_sum):
            return torch.zeros_like(y_sum)
        top_k_values, top_k_indices = torch.topk(y_sum, top_k, dim=1)
        out = torch.zeros_like(y_sum)
        out.scatter_(1, top_k_indices, 1)
        return out

    def __CA1_fire__(self, y_sum):
        """
        Fires CA1 neurons based on the input tensor.

        Args:
            y_sum (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after CA1 firing.
        """

        if self.CA1_WTA:
            top_k = int(self.mem_size*self.fq*0.5)
            spikes1 = self.__topk_thresh__(y_sum, top_k)
        else:
            spikes1 = (y_sum  >= self.thr_ca1).float()
        return spikes1

    def __one_shot_learning__(self, inputs):
        """
        Performs one-shot learning on the input.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after one-shot learning.
        """

        fq_half = self.fq / 2
        M = inputs.shape[0]

        if self.CA1_WTA:
            num_ones_per_row = int(self.mem_size * fq_half)
            plateau_winner1_dev0 = torch.zeros(M, 1, self.mem_size).cuda().float()
            for i in range(M):
                row_indices = torch.randperm(self.mem_size)[:num_ones_per_row]
                plateau_winner1_dev0[i, :, row_indices] = 1.
        else:
            plateau_winner1_dev0 = (torch.rand(M, 1, self.mem_size) <= fq_half).cuda().float()

        plateau_winner2_dev0 = (torch.rand(M, 1, self.mem_size) <= fq_half / (1 - fq_half)).cuda().float()
        plateau_winner2_dev0 = plateau_winner2_dev0 * (1 - plateau_winner1_dev0)  # mutual exclusive

        if self.parallel_btsp:

            plateau_winner1_dev0 = plateau_winner1_dev0.reshape(M,self.mem_size)
            # plateau_winner2_dev0 = plateau_winner2_dev0.reshape(M,self.mem_size)

            self.W_feed = (self.W_feed + inputs.T @ plateau_winner1_dev0) % 2
            y_sum = (self.W_feed.float() * self.W_feed_mask.float()).T @ inputs.T

            spikes1 = self.__CA1_fire__(y_sum.T).T


            self.W_back += spikes1 @ inputs
            self.W_back = (self.W_back >= 1).float()

            self.W_feed = self.W_feed.float() * self.W_feed_mask.float()
            self.W_back = self.W_back.float() * self.W_back_mask.float().T

            out = spikes1.T

            return out


    def __step_forward__(self, inputs):
        """
        Performs a forward step of the model.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after the forward step.
        """
        y_sum = (inputs @ self.W_feed)
        out = self.__CA1_fire__(y_sum)
        return out
    
    def __step_backward__(self, mem, topk=None):
        """
        Performs a backward step of the model.
        Args:
            mem (torch.Tensor): Memory tensor.
            topk (list, optional): List of top-k values for CA3_WTA. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after the backward step.
        """
        ca3_v = (mem @ self.W_back)
        if self.CA3_WTA:
            # assert topk != None, "topk should be set when CA3_WTA is True"
            if topk is None:
                topk = [int(self.input_size*self.fp)]
            out_seg = len(topk)
            ca3_v = torch.chunk(ca3_v, out_seg, dim=1)
            ca3_out = []
            for i in range(out_seg):
                ca3_out.append(self.__topk_thresh__(ca3_v[i], topk[i]))
            return torch.cat(ca3_out, dim=1)
        else:
            outs = (ca3_v  >= self.thr_ca3).float()
            return outs
    
    def forward_backward(self, inputs):
        """
        Performs a forward-backward pass of the model.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after the forward-backward pass.
        """
        mem = self.__step_forward__(inputs)
        rec = self.__step_backward__(mem)
        return rec


"""
Base binding model class.
Args:
    input_size (int): Size of the input.
    mem_size (int): Size of the memory.
    fq (float): Possibility hyperparameter for CA1_WTA.
    fp (float): Possibility hyperparameter for CA3_WTA.
    fw (float): Possibility hyperparameter for model learnable weights.
    L (int): Number of neurons in the output layer.
    K (int): Number of neurons in the binding layer.
    Nw (int): Number of neurons in the word layer.
    device (str, optional): Device to run the model on. Defaults to None.
Attributes:
    L (int): Number of neurons in the output layer.
    K (int): Number of neurons in the binding layer.
    Nw (int): Number of neurons in the word layer.
    model (BtspModel): BTSP model instance.
"""
class BaseBindingModel:

    def __init__(self, input_size, mem_size, fq, fp, fw, L, K, Nw, device=None):
        """
        Initializes the base binding model.
        Args:
            input_size (int): Size of the input.
            mem_size (int): Size of the memory.
            fq (float): Possibility hyperparameter for CA1_WTA.
            fp (float): Possibility hyperparameter for CA3_WTA.
            fw (float): Possibility hyperparameter for model learnable weights.
            L (int): Number of neurons in the output layer.
            K (int): Number of neurons in the binding layer.
            Nw (int): Number of neurons in the word layer.
            device (str, optional): Device to run the model on. Defaults to None.
        """
        
        # binding setting
        self.L = L
        self.K = K
        self.Nw = Nw

        self.model = BtspModel(input_size, mem_size, fq, fp, fw, device=device)

    
    def __binding_model_forward__(self, inputs):
        """
        Performs a forward step of the binding model.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after the forward step.
        """
        return self.model.__step_forward__(inputs)
    
    def __binding_model_backward__(self, CR):
        """
        Performs a backward step of the binding model.
        Args:
            CR (torch.Tensor): Composed representation tensor.
        Returns:
            torch.Tensor: Output tensor after the backward step.
        """
        return self.model.__step_backward__(CR)

"""
Flat binding model class.
"""
class FlatBindingModel(BaseBindingModel):

    def __init__(self, input_size, mem_size, fq, fp, fw, L, K, Nw, device=None):
        super().__init__(input_size, mem_size, fq, fp, fw, L, K, Nw, device=device)

    def binding(self, inputs):
        """
        Performs binding on the input.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Composed representation tensor.
        """
        cr = self.model.__one_shot_learning__(inputs)
        return cr
    
    def top_down_unbinding(self, composed_representation):
        """
        Performs top-down unbinding on the composed representation.
        Args:
            composed_representation (torch.Tensor): Composed representation tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """
        rec = self.__binding_model_backward__(composed_representation)
        return rec
    
    def bottom_up_unbinding(self, masked_input):
        """
        Performs bottom-up unbinding on the masked input.
        Args:
            masked_input (torch.Tensor): Masked input tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """
        cr = self.__binding_model_forward__(masked_input)
        rec = self.__binding_model_backward__(cr)
        return rec

    
"""
Hierarchical binding model class.
Attributes:
    depth (int): Depth of the hierarchical model.
"""
class HierarchicalBindingModel(BaseBindingModel):
    def __init__(self, input_size, mem_size, fq, fp, fw, L, K, Nw, device=None):
        super().__init__(input_size, mem_size, fq, fp, fw, L, K, Nw, device=device)

        self.depth = int(np.ceil(np.log2(K)))


    def __binding_model_forward__(self, inputs):
        """
        Performs a forward step of the hierarchical binding model.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Composed representation tensor.
        """
        words_list = torch.chunk(inputs, chunks=self.K, dim=1)
        num_at_first_level = 2 ** self.depth
        next_out_lists = words_list
        for d in range(self.depth):
            depth_items_num = int(num_at_first_level * (0.5 ** int(d)))
            next_depth_items_num = int(depth_items_num/2)
            out_lists = next_out_lists
            next_out_lists = []
            for i in range(next_depth_items_num):
                if (2*i+1) < np.ceil(self.K * 0.5 ** d):
                    left_vec = out_lists[2*i]
                    right_vec = out_lists[2*i+1]
                    inputs = torch.cat([left_vec, right_vec], dim=1)
                    outputs = self.model.__step_forward__(inputs)
                elif 2*i < np.ceil(self.K * 0.5 ** d):
                    left_vec = out_lists[2*i]
                    inputs = left_vec
                    outputs = inputs
                next_out_lists.append(outputs)
        composed_representation = outputs
        return composed_representation
    
    def __binding_model_backward__(self, composed_representation):
        """
        Performs a backward step of the hierarchical binding model.
        Args:
            composed_representation (torch.Tensor): Composed representation tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """
        next_mem = [composed_representation]
        num_at_each_level = {2: [1,2],
                                3: [1,2,3],
                                4: [1,2,4],
                                5: [1,2,3,5],
                                6: [1,2,3,6],
                                7: [1,2,4,7],
                                8: [1,2,4,8],
                            }

        for d in range(self.depth):
            if d == self.depth - 1:
                top_k = int(self.L*self.model.fp)
            else:
                top_k = int(self.L*self.model.fq/2)

            # depth_items_num = num_at_each_level[self.K][d]
            depth_items_num_binded = np.floor(num_at_each_level[self.K][d+1] / 2 )

            mem_list = next_mem
            next_mem = []
            for i in range(len(mem_list)):
                mem = mem_list[i]
                if i < depth_items_num_binded:
                    topk = [top_k,top_k]
                    curr_unbinded = self.model.__step_backward__(mem, topk=topk)
                    curr_unbinded = torch.chunk(curr_unbinded, 2, dim=1)
                    next_mem = next_mem + [curr_unbinded[0], curr_unbinded[1]]
                else:
                    next_mem = next_mem + [mem]

        assert len(next_mem) == self.K
        rec = torch.cat(next_mem[:self.K], dim=1)
        return rec
    
    def binding(self, inputs):
        """
        Performs binding on the input.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Composed representation tensor.
        """

        words_list = torch.chunk(inputs, chunks=self.K, dim=1)
        num_at_first_level = 2 ** self.depth
        next_out_lists = words_list

        for d in range(self.depth):
            depth_items_num = int(num_at_first_level * (0.5 ** int(d)))
            next_depth_items_num = int(depth_items_num/2)
            out_lists = next_out_lists
            next_out_lists = []
            for i in range(next_depth_items_num):
                if (2*i+1) < np.ceil(self.K * 0.5 ** d):
                    left_vec = out_lists[2*i]
                    right_vec = out_lists[2*i+1]
                    inputs = torch.cat([left_vec, right_vec], dim=1)
                    outputs = self.model.__one_shot_learning__(inputs)
                elif 2*i < np.ceil(self.K * 0.5 ** d):
                    left_vec = out_lists[2*i]
                    inputs = left_vec
                    outputs = inputs
                next_out_lists.append(outputs)
        composed_representation = outputs
        return composed_representation
    
    def top_down_unbinding(self, composed_representation):
        """
        Performs top-down unbinding on the composed representation.
        Args:
            composed_representation (torch.Tensor): Composed representation tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """

        return self.__binding_model_backward__(composed_representation)
    
    def bottom_up_unbinding(self, masked_input):
        """
        Performs bottom-up unbinding on the masked input.
        Args:
            masked_input (torch.Tensor): Masked input tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """

        cr = self.__binding_model_forward__(masked_input)
        rec = self.__binding_model_backward__(cr)
        return rec
    
"""
Online binding model class.
Attributes:
    depth (int): Depth of the online model.
"""
class OnlineBindingModel(BaseBindingModel):
    def __init__(self, input_size, mem_size, fq, fp, fw, L, K, Nw, device=None):
        super().__init__(input_size, mem_size, fq, fp, fw, L, K, Nw, device=device)

    def __binding_model_forward__(self, inputs):
        """
        Performs a forward step of the online binding model.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Composed representation tensor.
        """

        words_list = torch.chunk(inputs, self.K, dim=1)
        depth = int(self.K-1)
        for d in range(depth):
            if d == 0:
                inputs = torch.cat([words_list[0], words_list[1]], dim=1)
            else:
                inputs = torch.cat([outputs, words_list[d+1]], dim=1)
            outputs = self.model.__step_forward__(inputs)
        composed_representation = outputs
        return composed_representation
    
    def __binding_model_backward__(self, composed_representation):
        """
        Performs a backward step of the online binding model.
        Args:
            composed_representation (torch.Tensor): Composed representation tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """

        bind_depth = self.K - 1
        unbind_terms = []
        mem = composed_representation
        for d in range(bind_depth):
            if d != bind_depth-1:
                tokp = [int(self.L*self.model.fq/2), int(self.L*self.model.fp)]
            else:
                tokp = [int(self.L*self.model.fp), int(self.L*self.model.fp)]
            curr_unbinded = self.model.__step_backward__(mem, tokp)
            curr_unbinded = torch.chunk(curr_unbinded, 2, dim=1)
            next_mem = curr_unbinded[0]
            curr_unbind_term_ = curr_unbinded[1]
            unbind_terms.append(curr_unbind_term_)
            mem = next_mem
        unbind_terms.append(mem)
        rec = torch.cat(unbind_terms[::-1], dim=1)
        return rec
    
    
    def binding(self, inputs):
        """
        Performs binding on the input.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Composed representation tensor.
        """

        words_list = torch.chunk(inputs, chunks=self.K, dim=1)
        depth = int(self.K-1)
        for d in range(depth):
            if d == 0:
                inputs = torch.cat([words_list[0], words_list[1]], dim=1)
            else:
                inputs = torch.cat([outputs, words_list[d+1]], dim=1)
            outputs = self.model.__one_shot_learning__(inputs)

        composed_representation = outputs
        return composed_representation
        
    
    def top_down_unbinding(self, composed_representation):
        """
        Performs top-down unbinding on the composed representation.
        Args:
            composed_representation (torch.Tensor): Composed representation tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """

        return self.__binding_model_backward__(composed_representation)
    
    def bottom_up_unbinding(self, masked_input):
        """
        Performs bottom-up unbinding on the masked input.
        Args:
            masked_input (torch.Tensor): Masked input tensor.
        Returns:
            torch.Tensor: Reconstructed tensor after unbinding.
        """

        cr = self.__binding_model_forward__(masked_input)
        rec = self.__binding_model_backward__(cr)
        return rec
