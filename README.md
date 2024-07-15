# scheme-switching
FHE compiler to do scheme switching

list of operations: 
| Name                | Symbol | Supported by FHE Schemes   |
|---------------------|--------|----------------------------|
| Negation            | -      | TFHE, CKKS, BGV            |
| Addition            | +      | TFHE, CKKS, BGV            |
| Subtraction         | -      | TFHE, CKKS, BGV            |
| Multiplication      | *      | TFHE, CKKS, BGV            |
| Division            | /      | TFHE, CKKS                 |
| Remainder           | %      | TFHE, CKKS                 |
| Logical NOT         | !      | TFHE                       |
| Bitwise AND         | &      | TFHE                       |
| Bitwise OR          | \|     | TFHE                       |
| Bitwise XOR         | ^      | TFHE                       |
| Bitwise Shift Right | >>     | TFHE                       |
| Bitwise Shift Left  | <<     | TFHE                       |
| Minimum             | min    | TFHE, CKKS                 |
| Maximum             | max    | TFHE, CKKS                 |
| Greater Than        | gt     | TFHE, CKKS                 |
| Greater or Equal    | ge     | TFHE, CKKS                 |
| Less Than           | lt     | TFHE, CKKS                 |
| Less or Equal       | le     | TFHE, CKKS                 |
| Equal               | eq     | TFHE, CKKS, BGV            |
| Cast (into dest type)| cast_into | TFHE, CKKS, BGV        |
| Cast (from src type)| cast_from | TFHE, CKKS, BGV         |
| Ternary Operator    | select | TFHE                       |

