from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Define operations
operations = ['op2','op2','op2','op1', 'op3']  # Example: 4 operations
operation_types = {'op1': 'cmp', 'op2': 'mul', 'op3': 'add'}

# Execution times
execution_times = {
    'TFHE': {'cmp': 1, 'mul': 5, 'add': 1},
    'CKKS': {'cmp': 20, 'mul': 1, 'add': 1}
}

# Scheme switching cost (10 CKKS multiplications)
switch_cost = 7

# Create ILP problem
prob = LpProblem("FHE_Scheme_Selection", LpMinimize)

# Decision variables for schemes
scheme = {op: {s: LpVariable(f"x_{op}_{s}", 0, 1, cat="Binary") for s in ['TFHE', 'CKKS']} for op in operations}

# Decision variables for switching
switch = {i: LpVariable(f"switch_{i}", 0, 1, cat="Binary") for i in range(len(operations) - 1)}

# Objective: Minimize total cost
operation_cost = lpSum(
    scheme[op]['TFHE'] * execution_times['TFHE'][operation_types[op]] +
    scheme[op]['CKKS'] * execution_times['CKKS'][operation_types[op]]
    for op in operations
)
switch_cost_total = lpSum(switch[i] * switch_cost for i in range(len(operations) - 1))
prob += operation_cost + switch_cost_total

# Constraints
# 1. Each operation must be assigned to exactly one scheme
for op in operations:
    prob += scheme[op]['TFHE'] + scheme[op]['CKKS'] == 1

# 2. Enforce switching logic
for i in range(len(operations) - 1):
    # Define variables for consecutive operations
    op1, op2 = operations[i], operations[i + 1]
    prob += switch[i] >= scheme[op1]['TFHE'] - scheme[op2]['TFHE']
    prob += switch[i] >= scheme[op2]['TFHE'] - scheme[op1]['TFHE']

# Solve the problem
prob.solve()

# Output results
print("Optimal Operation Assignments:")
for op in operations:
    assigned_scheme = 'TFHE' if value(scheme[op]['TFHE']) == 1 else 'CKKS'
    print(f"  {op}: {assigned_scheme}")

print("\nSwitching Decisions:")
for i in range(len(operations) - 1):
    if value(switch[i]) == 1:
        print(f"  Switch between {operations[i]} and {operations[i + 1]}")
