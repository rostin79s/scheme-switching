#ifndef FHE_DAG_HPP
#define FHE_DAG_HPP

#include "dag.hpp"

class FHEDAG : public DAG {
public:
    FHEDAG() : DAG() {}

    void convertToFHEDAG(DAG *dag);
};

#endif // FHE_DAG_HPP
