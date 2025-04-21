import random
import time
from graphviz import Digraph

def visualize_full_bplus_tree(tree, filename="bplustree_full", highlight_nodes=None, label=None, filetype="pdf"):
    from graphviz import Digraph

    dot = Digraph()
    dot.attr(dpi='300')
    node_counter = [0]
    visited = set()

    def add_node(dot, node):
        if id(node) in visited:
            return node._viz_id
        visited.add(id(node))

        node_id = f"node{node_counter[0]}"
        node_counter[0] += 1

        keys_label = "|".join(str(k) for k in node.keys)
        fillcolor = "yellow" if highlight_nodes and node in highlight_nodes else ("lightblue" if node.is_leaf else "white")

        dot.node(node_id, f"<f0> {keys_label}", shape="record", style="filled", fillcolor=fillcolor)
        node._viz_id = node_id

        if not node.is_leaf:
            for i, child in enumerate(node.children):
                child_id = add_node(dot, child)
                dot.edge(node_id, child_id, label=f"P{i}")

        return node_id

    add_node(dot, tree.root)

    if label:
        dot.attr(label=label, fontsize="20")

    dot.render(filename, format=filetype, cleanup=True)
    print(f"Full B+ Tree visualization saved to {filename}.{filetype}")

# Visualize only a path or partial B+ Tree
def visualize_bplus_tree(tree, filename="bplustree", path=None, highlight_node=None, label=None):
    dot = Digraph()
    node_counter = [0]
    visited = set()

    def add_node(dot, node):
        if id(node) in visited:
            return None
        visited.add(id(node))

        node_id = f"node{node_counter[0]}"
        node_counter[0] += 1

        label = "|".join(str(k) for k in node.keys)
        fillcolor = "yellow" if node is highlight_node else ("lightblue" if node.is_leaf else "white")

        dot.node(node_id, f"<f0> {label}", shape="record", style="filled", fillcolor=fillcolor)
        node._viz_id = node_id  

        if not node.is_leaf:
            for i, child in enumerate(node.children):
                if path is None or child in path:
                    child_id = add_node(dot, child)
                    if child_id:
                        dot.edge(node_id, child_id, label=f"P{i}")

        return node_id

    if path is None:
        path = [tree.root]

    for node in path:
        add_node(dot, node)

    if label:
        dot.attr(label=label, fontsize="20")
    dot.render(filename, format='png', cleanup=True)

def trace_path_to_leaf(tree, key):
    current = tree.root
    path = [current]
    while not current.is_leaf:
        i = 0
        while i < len(current.keys) and key >= current.keys[i]:
            i += 1
        current = current.children[i]
        path.append(current)
    return path


def generate_records(min_val, max_val, total):
    unique_keys = set()
    records = []
    random.seed(time.time())

    while len(unique_keys) < total:
        key = random.randint(min_val, max_val)
        if key not in unique_keys:
            unique_keys.add(key)
            records.append(key)
    return records

# Group records into blocks for sparse B+ Trees
def create_blocks(records, block_size):
    records.sort()
    return [records[i:i+block_size] for i in range(0, len(records), block_size)]

class BPlusTreeNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []  # Points to blocks (sparse) or records (dense)
        self.next = None    # Leaf node linkage

class BPlusTree:
    def __init__(self, order, mode="dense"):
        self.root = BPlusTreeNode(is_leaf=True)
        self.order = order
        self.mode = mode  # "dense" or "sparse"

    def insert(self, key_or_block):
        if self.mode == "dense":
            key = key_or_block
            self._insert_dense(key)
        elif self.mode == "sparse":
            rep_key, block = key_or_block
            self._insert_sparse(rep_key, block)

    def _insert_dense(self, key):
        root = self.root
        if root.is_leaf and len(root.keys) < self.order:
            self._insert_into_leaf(root, key, key)
        else:
            if len(root.keys) >= self.order:
                new_root = BPlusTreeNode()
                new_root.is_leaf = False
                new_root.children.append(self.root)
                self._split_child(new_root, 0, self.root)
                self.root = new_root
            self._insert_non_full(self.root, key, key)

    def _insert_sparse(self, key, block):
        root = self.root
        if root.is_leaf and len(root.keys) < self.order:
            self._insert_into_leaf(root, key, block)
        else:
            if len(root.keys) >= self.order:
                new_root = BPlusTreeNode()
                new_root.is_leaf = False
                new_root.children.append(self.root)
                self._split_child(new_root, 0, self.root)
                self.root = new_root
            self._insert_non_full(self.root, key, block)

    def _insert_non_full(self, node, key, value):
        if node.is_leaf:
            self._insert_into_leaf(node, key, value)
            return

        index = len(node.keys) - 1
        while index >= 0 and key < node.keys[index]:
            index -= 1
        index += 1

        if len(node.children[index].keys) >= self.order:
            self._split_child(node, index, node.children[index])
            if key > node.keys[index]:
                index += 1

        self._insert_non_full(node.children[index], key, value)

    def _insert_into_leaf(self, leaf, key, value):
        i = 0
        while i < len(leaf.keys) and key > leaf.keys[i]:
            i += 1
        leaf.keys.insert(i, key)
        leaf.children.insert(i, value)

    
    def _split_child(self, parent, index, child):
        new_node = BPlusTreeNode(is_leaf=child.is_leaf)
        mid_index = self.order // 2

        # Store the key to promote before modifying anything
        if child.is_leaf:
            promote_key = new_node.keys = child.keys[mid_index:]
            new_node.children = child.children[mid_index:]
            child.keys = child.keys[:mid_index]
            child.children = child.children[:mid_index]
            new_node.next = child.next
            child.next = new_node
            parent.keys.insert(index, new_node.keys[0])  # promote first key of new_node
        else:
            promote_key = child.keys[mid_index]
            new_node.keys = child.keys[mid_index + 1:]
            new_node.children = child.children[mid_index + 1:]
            child.keys = child.keys[:mid_index]
            child.children = child.children[:mid_index + 1]
            parent.keys.insert(index, promote_key)

        parent.children.insert(index + 1, new_node)

    def delete(self, key):
        self._delete_recursive(self.root, key)
        if not self.root.is_leaf and len(self.root.keys) == 0:
            self.root = self.root.children[0]

    #Leaf Node:	directly delete the key or block
    #Internal Node:	find correct child → recursively delete → rebalance if necessary
    def _delete_recursive(self, node, key):
        if node.is_leaf:
            if key in node.keys:
                idx = node.keys.index(key)
                node.keys.pop(idx)
                node.children.pop(idx)
            elif self.mode == "sparse":
                for i, block in enumerate(node.children):
                    if isinstance(block, list) and key in block:
                        block.remove(key)
                        if len(block) == 0:
                            node.keys.pop(i)
                            node.children.pop(i)
                        break
            return

        idx = 0
        while idx < len(node.keys) and key >= node.keys[idx]:
            idx += 1

        child = node.children[idx]
        self._delete_recursive(child, key)

        min_keys = (self.order + 1) // 2 - 1
        if len(child.keys) < min_keys:
            self._rebalance(node, idx)

    def _rebalance(self, parent, index):
        child = parent.children[index]

        if index > 0:
            left = parent.children[index - 1]
            if len(left.keys) > (self.order + 1) // 2 - 1:
                if child.is_leaf:
                    child.keys.insert(0, left.keys.pop())
                    child.children.insert(0, left.children.pop())
                    parent.keys[index - 1] = child.keys[0]
                else:
                    child.keys.insert(0, parent.keys[index - 1])
                    parent.keys[index - 1] = left.keys.pop()
                    child.children.insert(0, left.children.pop())
                return

        if index < len(parent.children) - 1:
            right = parent.children[index + 1]
            if len(right.keys) > (self.order + 1) // 2 - 1:
                if child.is_leaf:
                    child.keys.append(right.keys.pop(0))
                    child.children.append(right.children.pop(0))
                    parent.keys[index] = right.keys[0]
                else:
                    child.keys.append(parent.keys[index])
                    parent.keys[index] = right.keys.pop(0)
                    child.children.append(right.children.pop(0))
                return

        if index > 0:
            left = parent.children[index - 1]
            self._merge_nodes(parent, index - 1, left, child)
        else:
            right = parent.children[index + 1]
            self._merge_nodes(parent, index, child, right)

    def _merge_nodes(self, parent, index, left, right):
        if left.is_leaf:
            left.keys.extend(right.keys)
            left.children.extend(right.children)
            left.next = right.next
        else:
            left.keys.append(parent.keys[index])
            left.keys.extend(right.keys)
            left.children.extend(right.children)
        parent.keys.pop(index)
        parent.children.pop(index + 1)


    # Used for testing print small trees 
    def print_tree(self):
        queue = [(self.root, 0)]
        current_level = 0
        print("B+ Tree Structure:\n")

        while queue:
            next_queue = []
            print(f"Level {current_level}:")

            for node, level in queue:
                node_type = "Leaf" if node.is_leaf else "Internal"
                print(f"  {node_type} Node: Keys = {node.keys}")

                if node.is_leaf:
                    for i, key in enumerate(node.keys):
                        data = node.children[i]
                        print(f"    {key} → {data}")
                    if node.next:
                        print(f"    → Next leaf: {node.next.keys}")
                else:
                    for i, child in enumerate(node.children):
                        print(f"    P{i} → {child.keys}")
                        next_queue.append((child, level + 1))

            print()
            queue = next_queue
            current_level += 1

    def search(self, key):
        current = self.root

        while not current.is_leaf:
            i = 0
            while i < len(current.keys) and key >= current.keys[i]:
                i += 1
            current = current.children[i]

        # Linear scan at the leaf
        for i, k in enumerate(current.keys):
            if k == key:
                return current.children[i]  # dense: record; sparse: block
            elif self.mode == "sparse" and isinstance(current.children[i], list):
                # Check within block for sparse mode
                if key in current.children[i]:
                    return current.children[i]
        return None
    

    def range_search(self, start, end):
        result = []
        current = self.root

        # Find the first relevant leaf
        while not current.is_leaf:
            i = 0
            while i < len(current.keys) and start >= current.keys[i]:
                i += 1
            current = current.children[i]

        # Traverse leaf nodes using `next` pointer
        while current:
            for i, key in enumerate(current.keys):
                if start <= key <= end:
                    val = current.children[i]
                    if self.mode == "sparse" and isinstance(val, list):
                        result.extend([x for x in val if start <= x <= end])
                    else:
                        result.append(val)
                elif key > end:
                    return result
            current = current.next

        return result


# Step (a): Generate 10,000 unique records
records = generate_records(100000, 200000, 10000)

# Step (b): Create 4 B+ Trees
dense_13 = BPlusTree(order=13, mode="dense")
sparse_13 = BPlusTree(order=13, mode="sparse")
dense_24 = BPlusTree(order=24, mode="dense")
sparse_24 = BPlusTree(order=24, mode="sparse")

# Sparse needs blocks
def build_sparse_tree(tree, records, block_size):
    blocks = create_blocks(records, block_size)
    for block in blocks:
        tree.insert((block[0], block))
    return blocks

# Insert all records
for r in records:
    dense_13.insert(r)
    dense_24.insert(r)

sparse_13_blocks = build_sparse_tree(sparse_13, records, 13)
sparse_24_blocks = build_sparse_tree(sparse_24, records, 13)

visualize_full_bplus_tree(sparse_13, "sparse13_full", label="Full Sparse-13 B+ Tree", filetype="pdf")
visualize_full_bplus_tree(sparse_24, "sparse24_full", label="Full Sparse-24 B+ Tree", filetype="pdf")
visualize_full_bplus_tree(dense_13, "dense13_full", label="Dense-13 Full Tree", filetype="pdf")
visualize_full_bplus_tree(dense_24, "dense24_full", label="Dense-24 Full Tree", filetype="pdf")

# Step (c1): 2 random inserts on each dense tree
print("\n(c1) Random inserts to dense trees")
for i in range(2):
    val = random.randint(100000, 200000)

    # Dense 13
    path_before = trace_path_to_leaf(dense_13, val)
    visualize_bplus_tree(dense_13, f"dense13_insert_before_{i}", path=path_before, label=f"Insert {val} (before)")
    dense_13.insert(val)
    path_after = trace_path_to_leaf(dense_13, val)
    visualize_bplus_tree(dense_13, f"dense13_insert_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Insert {val} (after)")

    # Dense 24
    path_before = trace_path_to_leaf(dense_24, val)
    visualize_bplus_tree(dense_24, f"dense24_insert_before_{i}", path=path_before, label=f"Insert {val} (before)")
    dense_24.insert(val)
    path_after = trace_path_to_leaf(dense_24, val)
    visualize_bplus_tree(dense_24, f"dense24_insert_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Insert {val} (after)")

    print(f"Inserted {val} into dense trees")

# Step (c2): 2 random deletes on each sparse tree
print("\n(c2) Random deletes from sparse trees")
for i in range(2):
    key = random.choice([block[0] for block in sparse_13_blocks])  # choose actual keys used in insert()

    if not sparse_13.search(key):
        print(f"Key {key} not found in sparse_13, skipping delete")

    # Sparse 13
    path_before = trace_path_to_leaf(sparse_13, key)
    visualize_bplus_tree(sparse_13, f"sparse13_delete_before_{i}", path=path_before, label=f"Delete {key} (before)")
    sparse_13.delete(key)
    path_after = trace_path_to_leaf(sparse_13, key)
    visualize_bplus_tree(sparse_13, f"sparse13_delete_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Delete {key} (after)")

    print(f"Deleted {key} from sparse 13 tree")

    key = random.choice([block[0] for block in sparse_24_blocks])

    # Sparse 24
    path_before = trace_path_to_leaf(sparse_24, key)
    visualize_bplus_tree(sparse_24, f"sparse24_delete_before_{i}", path=path_before, label=f"Delete {key} (before)")
    sparse_24.delete(key)
    path_after = trace_path_to_leaf(sparse_24, key)
    visualize_bplus_tree(sparse_24, f"sparse24_delete_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Delete {key} (after)")

    print(f"Deleted {key} from sparse 24 tree")


# Step (c3): 5 more random insert/delete ops
print("\n(c3) 5 more random insert/delete ops")
for i in range(5):
    val = random.randint(100000, 200000)
    del_key13 = random.choice([block[0] for block in sparse_13_blocks])
    del_key24 = random.choice([block[0] for block in sparse_24_blocks])

    # Dense 13 insert
    path_before = trace_path_to_leaf(dense_13, val)
    visualize_bplus_tree(dense_13, f"dense13_mix_insert_before_{i}", path=path_before, label=f"Insert {val} (before)")
    dense_13.insert(val)
    path_after = trace_path_to_leaf(dense_13, val)
    visualize_bplus_tree(dense_13, f"dense13_mix_insert_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Insert {val} (after)")

    # Dense 24 insert
    path_before = trace_path_to_leaf(dense_24, val)
    visualize_bplus_tree(dense_24, f"dense24_mix_insert_before_{i}", path=path_before, label=f"Insert {val} (before)")
    dense_24.insert(val)
    path_after = trace_path_to_leaf(dense_24, val)
    visualize_bplus_tree(dense_24, f"dense24_mix_insert_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Insert {val} (after)")

    # Sparse 13 delete
    if sparse_13.search(del_key13):
        path_before = trace_path_to_leaf(sparse_13, del_key13)
        visualize_bplus_tree(sparse_13, f"sparse13_mix_delete_before_{i}", path=path_before, label=f"Delete {del_key13} (before)")
        sparse_13.delete(del_key13)
        path_after = trace_path_to_leaf(sparse_13, del_key13)
        visualize_bplus_tree(sparse_13, f"sparse13_mix_delete_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Delete {del_key13} (after)")
        print(f"Deleted {del_key13} from sparse_13")
    else:
        print(f"Key {del_key13} not found in sparse_13")

    # Sparse 24 delete
    if sparse_24.search(del_key24):
        path_before = trace_path_to_leaf(sparse_24, del_key24)
        visualize_bplus_tree(sparse_24, f"sparse24_mix_delete_before_{i}", path=path_before, label=f"Delete {del_key24} (before)")
        sparse_24.delete(del_key24)
        path_after = trace_path_to_leaf(sparse_24, del_key24)
        visualize_bplus_tree(sparse_24, f"sparse24_mix_delete_after_{i}", path=path_after, highlight_node=path_after[-1], label=f"Delete {del_key24} (after)")
        print(f"Deleted {del_key24} from sparse_24")
    else:
        print(f"Key {del_key24} not found in sparse_24")

    print(f"Inserted {val} into both dense trees")


# Step (c4): 5 random search ops
print("\n(c4) 5 random point searches")
for _ in range(5):
    target = random.choice(records)
    print(f"Search {target}:")
    print(f"  Dense 13 → {dense_13.search(target)}")
    print(f"  Sparse 13 → {sparse_13.search(target)}")
    print(f"  Dense 24 → {dense_24.search(target)}")
    print(f"  Sparse 24 → {sparse_24.search(target)}")

print("\n(c4b) 5 random range searches")
for _ in range(5):
    # pick low and high bounds (small window for readability)
    low  = random.randint(100000, 200000)
    high = random.randint(low, low + 500)
    print(f"Range Search [{low}, {high}]:")
    print(f"  Dense 13  → {dense_13.range_search(low, high)}")
    print(f"  Sparse 13 → {sparse_13.range_search(low, high)}")
    print(f"  Dense 24  → {dense_24.range_search(low, high)}")
    print(f"  Sparse 24 → {sparse_24.range_search(low, high)}")
