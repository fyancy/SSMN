import torch


def compute_logits(cluster_centers, data):
    """Computes the logits of being in one cluster, squared Euclidean.
    Args:
    cluster_centers: [K, D] Cluster center representation.
    data: [B, N, D] Data representation.
    Returns:
    log_prob: [B, N, K] logits.
    """
    k = cluster_centers.shape[0]
    b = data.shape[0]
    cluster_centers = cluster_centers.unsqueeze(dim=0)  # [1, K, D]
    data = data.contiguous().view(-1, data.shape[-1]).unsqueeze(dim=1)  # [N, 1, D]
    # neg_dist = -torch.sum(torch.pow(data - cluster_centers, 2), dim=-1)  # [N, K]
    neg_dist = -torch.mean(torch.pow(data - cluster_centers, 2), dim=-1)  # [N, K]
    neg_dist = neg_dist.view(b, -1, k)
    return neg_dist


def assign_cluster(cluster_centers, data):
    """Assigns data to cluster center, using K-Means.return the probability.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    data: [B, N, D] Data representation.
  Returns:
    prob: [B, N, K] Soft assignment.
  """
    logits = compute_logits(cluster_centers, data)  # [B, N, K]
    prob = torch.nn.functional.softmax(logits, dim=-1)
    # print(prob.cpu().detach().numpy())
    return prob


def update_cluster(data, prob):
    """Updates cluster center based on assignment, standard K-Means.
  Args:
    data: [B, N, D]. Data representation.
    prob: [B, N, K]. Cluster assignment soft probability.
  Returns:
    cluster_centers: [K, D]. Cluster center representation.
  """
    # Normalize across N.
    data = data.view(-1, data.shape[-1])  # [B*N, D]
    prob = prob.view(-1, prob.shape[-1])  # [B*N, K]
    prob_sum = torch.sum(prob, dim=0, keepdim=True)  # [1, K]
    prob_sum += torch.eq(prob_sum, 0.0).float()  # 防止某一维为0，作为分母
    prob2 = prob / prob_sum  # [B*N, K]/[1, K] ==> [B*N, K]  broadcast mechanism
    # sum instead of mean, easier to converge.
    cluster_centers = torch.sum(data.unsqueeze(dim=1) * prob2.unsqueeze(dim=2), dim=0)
    # sum([B*N, 1, D]*[B*N, K, 1])==>sum([B*N, K, D])==>[K, D]
    # print('---------cluster---------\n', cluster_centers.cpu().detach().numpy())

    return cluster_centers
