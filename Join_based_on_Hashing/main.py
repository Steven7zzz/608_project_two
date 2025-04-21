import random
import pandas as pd
from IPython.display import display

BLOCK_SIZE = 8
MAX_MEMORY_BLOCKS = 15
NUM_PARTITIONS = MAX_MEMORY_BLOCKS - 1

# ---------- Part 1 ----------

virtual_disk_S = []

# Generate 5000 unique B values
B_values = random.sample(range(10000, 50001), 5000)
S = []

for b in B_values:
    c = random.randint(1, 100000)  # C can be anything
    S.append((b, c))

# Store in blocks of size 8
for i in range(0, len(S), BLOCK_SIZE):
    block = S[i:i+BLOCK_SIZE]
    virtual_disk_S.append(block)


# ---------- Part 2 ----------

disk_read_count = 0
disk_write_count = 0
virtual_memory = []

def reset_memory():
    global virtual_memory
    virtual_memory = []

def disk_read(disk, block_id):
    global disk_read_count, virtual_memory
    if len(virtual_memory) >= MAX_MEMORY_BLOCKS:
        raise MemoryError("Memory overflow on read")
    block = disk[block_id]
    virtual_memory.append(block)
    disk_read_count += 1
    return block

def disk_write(disk, block, position=None):
    global disk_write_count
    if position is None:
        disk.append(list(block))
    else:
        disk[position] = list(block)
    disk_write_count += 1

# ---------- Part 3 ----------
def hash_b(b):
    return b % NUM_PARTITIONS

def one_pass_hash_join(R_disk, S_disk):
    global disk_read_count, virtual_memory
    reset_memory()
    disk_read_count = 0

    # Build hash table for S
    hash_tab = {}
    for i in range(len(S_disk)):
        block = disk_read(S_disk, i)
        for (b, c) in block:
            hash_tab.setdefault(b, []).append(c)
            
    # Stream R
    results = []
    for i in range(len(R_disk)):
        r_block = disk_read(R_disk, i)
        for (a, b) in r_block:
            for c in hash_tab.get(b, []):
                results.append((a, b, c))
    return results


# ---------- Part 4 ----------
def partition_phase(R_disk, S_disk):
    global disk_read_count, disk_write_count, virtual_memory
    reset_memory()
    disk_read_count = 0
    disk_write_count = 0

    # Prepare partitions
    R_parts = [[] for _ in range(NUM_PARTITIONS)]
    S_parts = [[] for _ in range(NUM_PARTITIONS)]
    bufR = {i: [] for i in range(NUM_PARTITIONS)}
    bufS = {i: [] for i in range(NUM_PARTITIONS)}

    # Partition R
    for i in range(len(R_disk)):
        block = disk_read(R_disk, i)
        # immediately free the block no memory overflow
        virtual_memory.pop()  

        for (a, b) in block:
            idx = hash_b(b)
            bufR[idx].append((a, b))
            if len(bufR[idx]) == BLOCK_SIZE:
                disk_write(R_parts[idx], bufR[idx])
                bufR[idx].clear()

    for i in range(NUM_PARTITIONS):
        if bufR[i]:
            disk_write(R_parts[i], bufR[i])
            bufR[i].clear()

    # Partition S
    for i in range(len(S_disk)):
        block = disk_read(S_disk, i)
        virtual_memory.pop()

        for (b, c) in block:
            idx = hash_b(b)
            bufS[idx].append((b, c))
            if len(bufS[idx]) == BLOCK_SIZE:
                disk_write(S_parts[idx], bufS[idx])
                bufS[idx].clear()

    for i in range(NUM_PARTITIONS):
        if bufS[i]:
            disk_write(S_parts[i], bufS[i])
            bufS[i].clear()

    return R_parts, S_parts

def two_pass_hash_join(R_disk, S_disk):
    global disk_read_count, virtual_memory
    reset_memory()
    disk_read_count = 0
    R_parts, S_parts = partition_phase(R_disk, S_disk)
    results = []

    for i in range(NUM_PARTITIONS):
        reset_memory()
        ht = {}

        for j in range(len(S_parts[i])):
            block = disk_read(S_parts[i], j)
            virtual_memory.pop()           

            for (b, c) in block:
                ht.setdefault(b, []).append(c)

        for j in range(len(R_parts[i])):
            block = disk_read(R_parts[i], j)
            virtual_memory.pop()           

            for (a, b) in block:
                for c in ht.get(b, []):
                    results.append((a, b, c))
    return results

# ---------- Part 5 ----------

# ——— Experiment 5.1 ——————————————————————————————————————————————
# Generate R1(A, B) with B from S (with replacement)
NUM_TUPLES_R1 = 1000
R1 = [(f"A{i}", random.choice(B_values)) for i in range(NUM_TUPLES_R1)]
R1_disk = [R1[i:i+8] for i in range(0, len(R1), 8)]


# Reset I/O counters and memory
reset_memory()
disk_read_count = 0
disk_write_count = 0

# Choose join method (virtual_disk_S is large, so two-pass)
joined_1 = two_pass_hash_join(R1_disk, virtual_disk_S)
io_counts_1 = {'reads': disk_read_count, 'writes': disk_write_count}

print("Matched joins before filter", len(joined_1))

# Pick 20 B-values to filter
picked_B = random.sample(B_values, 20)
print("Randomly picked B-values:", picked_B)

filtered_1 = [t for t in joined_1 if t[1] in picked_B]
df1 = pd.DataFrame(filtered_1, columns=['A', 'B', 'C'])

print("Experiment 5.1: Filtered Join Results")
print(df1.to_string(index=False))

print("Experiment 5.1 I/O counts:", io_counts_1)


# ——— Experiment 5.2 ——————————————————————————————————————————————

R2 = [(f"A{i}", random.randint(20000, 30000)) for i in range(1200)]
R2_disk = [R2[i:i+8] for i in range(0, len(R2), 8)]

# Reset I/O counters and memory
reset_memory()
disk_read_count = 0
disk_write_count = 0

joined_2 = two_pass_hash_join(R2_disk, virtual_disk_S)
io_counts_2 = {'reads': disk_read_count, 'writes': disk_write_count}

df2 = pd.DataFrame(joined_2, columns=['A', 'B', 'C'])

print("Experiment 5.2: Complete Join Results")

print("Matched joins", len(joined_2))

print(df2.to_string(index=False))

print("Experiment 5.2 I/O counts:", io_counts_2)