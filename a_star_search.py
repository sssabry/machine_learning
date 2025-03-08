adjacency = {
    # for each: 'node': [{neighbour: edgeCost}]
    'S': {'A': 4, 'C': 11},
    'C': {'E': 2, 'H': 4},
    'A': {'D': 5, 'B': 8},
    'B': {'D': 3, 'E': 12},
    'D': {'E': 8},
    'E': {'G': 7, 'F': 3},
    'H': {'E': 6, 'G': 1},
    'G': {},
    'F': {'G': 9}
}

heuristics = {
    'S': 7,
    'C': 5,
    'A': 8,
    'B': 6,
    'D': 5,
    'E': 3,
    'H': 7,
    'G': 0,
    'F': 3
}


def astar(start, target):
    # tracks open nodes, each node -> (node, f(n)_score)
    l1 = []
    l1.append((start, 0 + heuristics[start]))

    l2 = []
    g_scores = {start: 0}  # track the updated g_scores as we go

    while l1:  # not empty

        # find node with lowest f(n) in l1
        lowest_fn = float('inf')
        lowest_node = None

        for node, fn in l1:
            if fn < lowest_fn:
                lowest_fn = fn
                lowest_node = node

        # found it: remove it from l1
        l1.remove((lowest_node, lowest_fn))

        # if node is goal -> return
        if lowest_node == target:
            print("Target node reached.")
            return True

        l2.append(lowest_node)

        # compute f(neighbours)
        for neighbour, edgecost in adjacency[lowest_node].items():
            # compute f(n)
            # up until current node + edge cost
            new_gn = g_scores[lowest_node] + edgecost
            new_fn = new_gn + heuristics[neighbour]

            # make booleans to check if its in either
            in_l1 = any(n == neighbour for n, _ in l1)

            # if neighbour not in l1 then add it:
            if not in_l1:
                l1.append((neighbour, new_fn))  # add to l1
                g_scores[neighbour] = new_gn  # update g_score with travel cost

    print(f"Target node {target} not reached from {start}")
    return False


astar('S', 'G')
