#include "fhedag.hpp"

void FHEDAG::convertToFHEDAG(DAG *dag) {

    // Iterate over all nodes in the original DAG
    for (DAGNode *node : dag->nodes) {
        std::string op = node->operation;
        std::vector<std::string> ops = node->operands;
        std::string type = node->operandType;

        bool isCiphertextOperation = false;
        bool hasPlaintext = false;

        // Determine if the operation should be a FHE operation based on its operands
        for (const auto &operand : ops) {
            if (dag->functionInputs.find(operand) != dag->functionInputs.end()) {
                isCiphertextOperation = true;
            } else {
                hasPlaintext = true;
            }
        }

        // Convert the operation name accordingly
        if (isCiphertextOperation) {
            if (hasPlaintext) {
                op = "FHE" + op + "P";
            } else {
                op = "FHE" + op;
            }
        }

        // Add the node to the FHE DAG
        DAGNode *fheNode = addNode(node->result, op, ops, type);

        // Maintain dependencies (edges)
        for (DAGNode *dep : node->dependencies) {
            addEdge(dep, fheNode);
        }
    }
}
