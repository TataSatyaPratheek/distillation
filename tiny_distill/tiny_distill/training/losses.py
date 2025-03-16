"""
Loss functions for knowledge distillation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean"
) -> torch.Tensor:
    """
    Knowledge distillation loss via KL divergence.
    
    Args:
        student_logits (torch.Tensor): Student model logits
        teacher_logits (torch.Tensor): Teacher model logits
        temperature (float, optional): Softmax temperature. Defaults to 2.0.
        reduction (str, optional): Reduction method. Defaults to "batchmean".
        
    Returns:
        torch.Tensor: KL divergence loss
    """
    # Apply temperature scaling
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    
    # Compute log softmax for student and softmax for teacher
    log_probs_student = F.log_softmax(student_logits_scaled, dim=-1)
    probs_teacher = F.softmax(teacher_logits_scaled, dim=-1)
    
    # Compute KL divergence
    kl_loss = F.kl_div(log_probs_student, probs_teacher, reduction=reduction)
    
    # Scale by temperature squared (as in the original paper)
    return kl_loss * (temperature ** 2)


def masked_kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Knowledge distillation loss with masking for padding tokens.
    
    Args:
        student_logits (torch.Tensor): Student model logits [batch_size, seq_len, vocab_size]
        teacher_logits (torch.Tensor): Teacher model logits [batch_size, seq_len, vocab_size]
        attention_mask (Optional[torch.Tensor], optional): Attention mask [batch_size, seq_len]. Defaults to None.
        temperature (float, optional): Softmax temperature. Defaults to 2.0.
        
    Returns:
        torch.Tensor: Masked KL divergence loss
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Apply temperature scaling
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    
    # Compute log softmax for student and softmax for teacher
    log_probs_student = F.log_softmax(student_logits_scaled, dim=-1)
    probs_teacher = F.softmax(teacher_logits_scaled, dim=-1)
    
    # Compute KL divergence (per token)
    # kl_div expects input in log-space and target in probability space
    kl_div = F.kl_div(log_probs_student.reshape(-1, vocab_size),
                       probs_teacher.reshape(-1, vocab_size),
                       reduction="none").sum(-1)
    
    # Reshape back to [batch_size, seq_len]
    kl_div = kl_div.reshape(batch_size, seq_len)
    
    # Apply mask if provided
    if attention_mask is not None:
        # Ensure mask is binary and same device
        mask = attention_mask.to(dtype=torch.bool, device=kl_div.device)
        
        # Apply mask and compute mean over non-masked tokens
        kl_div = kl_div * mask
        num_tokens = mask.sum()
        
        # Avoid division by zero
        if num_tokens > 0:
            kl_div = kl_div.sum() / num_tokens
        else:
            kl_div = kl_div.sum()  # Will be zero if all tokens are masked
    else:
        # No mask, compute mean over all tokens
        kl_div = kl_div.mean()
    
    # Scale by temperature squared (as in the original paper)
    return kl_div * (temperature ** 2)


def mse_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Mean squared error loss between student and teacher logits.
    
    Args:
        student_logits (torch.Tensor): Student model logits
        teacher_logits (torch.Tensor): Teacher model logits
        attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
        
    Returns:
        torch.Tensor: MSE loss
    """
    if attention_mask is not None:
        # Apply mask (expand to match logits dimension)
        mask = attention_mask.unsqueeze(-1).expand_as(student_logits)
        
        # Compute masked MSE
        squared_error = mask * (student_logits - teacher_logits) ** 2
        
        # Compute mean over non-masked values
        num_elements = mask.sum()
        if num_elements > 0:
            return squared_error.sum() / num_elements
        else:
            return torch.tensor(0.0, device=student_logits.device)
    else:
        # Standard MSE
        return F.mse_loss(student_logits, teacher_logits)


def attention_transfer_loss(
    student_attentions: List[torch.Tensor],
    teacher_attentions: List[torch.Tensor]
) -> torch.Tensor:
    """
    Attention transfer loss to match attention patterns.
    
    Args:
        student_attentions (List[torch.Tensor]): Student attention matrices
        teacher_attentions (List[torch.Tensor]): Teacher attention matrices
        
    Returns:
        torch.Tensor: Attention transfer loss
    """
    # Ensure we have same number of layers to compare
    assert len(student_attentions) == len(teacher_attentions), \
        f"Mismatched number of attention layers: {len(student_attentions)} vs {len(teacher_attentions)}"
    
    loss = 0.0
    for student_attn, teacher_attn in zip(student_attentions, teacher_attentions):
        # Handle different attention shapes (e.g., different number of heads)
        if student_attn.shape != teacher_attn.shape:
            # Average over the head dimension
            student_attn_avg = student_attn.mean(dim=1)  # [batch_size, seq_len, seq_len]
            teacher_attn_avg = teacher_attn.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Compute Frobenius norm of difference
            loss += F.mse_loss(student_attn_avg, teacher_attn_avg)
        else:
            # Direct comparison
            loss += F.mse_loss(student_attn, teacher_attn)
    
    # Average over all layers
    return loss / len(student_attentions)


def hidden_state_transfer_loss(
    student_hiddens: List[torch.Tensor],
    teacher_hiddens: List[torch.Tensor]
) -> torch.Tensor:
    """
    Hidden state transfer loss to match model activations.
    
    Args:
        student_hiddens (List[torch.Tensor]): Student hidden states
        teacher_hiddens (List[torch.Tensor]): Teacher hidden states
        
    Returns:
        torch.Tensor: Hidden state transfer loss
    """
    # Get subset of layers to compare (select every N layers from teacher to match student)
    num_student_layers = len(student_hiddens)
    num_teacher_layers = len(teacher_hiddens)
    
    # Sample teacher layers to match student layers
    teacher_layer_indices = [int(i * num_teacher_layers / num_student_layers) for i in range(num_student_layers)]
    sampled_teacher_hiddens = [teacher_hiddens[i] for i in teacher_layer_indices]
    
    loss = 0.0
    for i, (student_hidden, teacher_hidden) in enumerate(zip(student_hiddens, sampled_teacher_hiddens)):
        # Handle different hidden state dimensions
        if student_hidden.shape != teacher_hidden.shape:
            # Project teacher to student dimension using simple averaging
            if student_hidden.shape[-1] < teacher_hidden.shape[-1]:
                # Reduce teacher dimension by averaging
                ratio = teacher_hidden.shape[-1] // student_hidden.shape[-1]
                teacher_hidden_resized = teacher_hidden.reshape(*teacher_hidden.shape[:-1], 
                                                              student_hidden.shape[-1], ratio)
                teacher_hidden_resized = teacher_hidden_resized.mean(dim=-1)
            else:
                # Increase teacher dimension by repeating (not ideal but simple)
                ratio = student_hidden.shape[-1] // teacher_hidden.shape[-1]
                teacher_hidden_resized = teacher_hidden.unsqueeze(-1).repeat(1, 1, 1, ratio)
                teacher_hidden_resized = teacher_hidden_resized.reshape(*teacher_hidden.shape[:-1], 
                                                                    student_hidden.shape[-1])
            
            # Compute MSE
            loss += F.mse_loss(student_hidden, teacher_hidden_resized)
        else:
            # Direct comparison
            loss += F.mse_loss(student_hidden, teacher_hidden)
    
    # Average over all layers
    return loss / num_student_layers


def combined_distillation_loss(
    student_outputs: Dict[str, Any],
    teacher_outputs: Dict[str, Any],
    temperature: float = 2.0,
    alpha: float = 0.5,
    beta: float = 0.0,
    gamma: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss function for knowledge distillation.
    
    Args:
        student_outputs (Dict[str, Any]): Student model outputs
        teacher_outputs (Dict[str, Any]): Teacher model outputs
        temperature (float, optional): Temperature for KL divergence. Defaults to 2.0.
        alpha (float, optional): Weight for KL divergence loss. Defaults to 0.5.
        beta (float, optional): Weight for attention transfer loss. Defaults to 0.0.
        gamma (float, optional): Weight for hidden state loss. Defaults to 0.0.
        attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
        
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and individual loss components
    """
    losses = {}
    
    # KL divergence loss on logits
    if alpha > 0:
        losses["kl_loss"] = masked_kl_divergence_loss(
            student_logits=student_outputs["logits"],
            teacher_logits=teacher_outputs["logits"],
            attention_mask=attention_mask,
            temperature=temperature
        )
    else:
        losses["kl_loss"] = torch.tensor(0.0, device=student_outputs["logits"].device)
    
    # Attention transfer loss
    if beta > 0 and "attentions" in student_outputs and "attentions" in teacher_outputs:
        losses["attention_loss"] = attention_transfer_loss(
            student_attentions=student_outputs["attentions"],
            teacher_attentions=teacher_outputs["attentions"]
        )
    else:
        losses["attention_loss"] = torch.tensor(0.0, device=student_outputs["logits"].device)
    
    # Hidden state transfer loss
    if gamma > 0 and "hidden_states" in student_outputs and "hidden_states" in teacher_outputs:
        losses["hidden_loss"] = hidden_state_transfer_loss(
            student_hiddens=student_outputs["hidden_states"],
            teacher_hiddens=teacher_outputs["hidden_states"]
        )
    else:
        losses["hidden_loss"] = torch.tensor(0.0, device=student_outputs["logits"].device)
    
    # Combine losses
    total_loss = alpha * losses["kl_loss"] + beta * losses["attention_loss"] + gamma * losses["hidden_loss"]
    
    return total_loss, losses


class DistillationLoss(nn.Module):
    """Loss module for knowledge distillation."""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.0,
        gamma: float = 0.0
    ):
        """
        Initialize distillation loss.
        
        Args:
            temperature (float, optional): Temperature for KL divergence. Defaults to 2.0.
            alpha (float, optional): Weight for KL divergence loss. Defaults to 0.5.
            beta (float, optional): Weight for attention transfer loss. Defaults to 0.0.
            gamma (float, optional): Weight for hidden state loss. Defaults to 0.0.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(
        self,
        student_outputs: Union[Dict[str, Any], torch.Tensor],
        teacher_outputs: Union[Dict[str, Any], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            student_outputs (Union[Dict[str, Any], torch.Tensor]): Student model outputs
            teacher_outputs (Union[Dict[str, Any], torch.Tensor]): Teacher model outputs
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]: Loss or loss with components
        """
        # Handle tensor inputs (logits only)
        if isinstance(student_outputs, torch.Tensor) and isinstance(teacher_outputs, torch.Tensor):
            student_outputs = {"logits": student_outputs}
            teacher_outputs = {"logits": teacher_outputs}
        
        total_loss, losses = combined_distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            temperature=self.temperature,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            attention_mask=attention_mask
        )
        
        return total_loss, losses


class CachedDistillationLoss(nn.Module):
    """Loss module for distillation with cached teacher outputs."""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 1.0
    ):
        """
        Initialize cached distillation loss.
        
        Args:
            temperature (float, optional): Temperature for KL divergence. Defaults to 2.0.
            alpha (float, optional): Weight for KL divergence loss. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            student_logits (torch.Tensor): Student model logits
            teacher_logits (torch.Tensor): Cached teacher model logits
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            torch.Tensor: Distillation loss
        """
        # Ensure teacher logits are on the same device as student logits
        if teacher_logits.device != student_logits.device:
            teacher_logits = teacher_logits.to(student_logits.device)
        
        # Compute KL divergence loss
        kl_loss = masked_kl_divergence_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
            temperature=self.temperature
        )
        
        return self.alpha * kl_loss