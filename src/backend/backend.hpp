#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "../frontend/dag.hpp"


void generateCPP(const DAG& dag);
void printHeaders(std::ofstream& IR);
void printUserFunction(std::ofstream& IR, const DAG& dag);
void printMainFunction(std::ofstream& IR, const DAG& dag);
#endif