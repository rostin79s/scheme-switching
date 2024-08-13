#ifndef CTXT_HPP
#define CTXT_HPP



class CFHEctxt {
  public:
    
    // seal::Ciphertext ctxt;
    // seal::scheme_type scheme;
    int noise_budget;
    int mult_depth;
    int add_depth;
    double scale;
};

class TFHEctxt {
  public:
    
    // seal::Ciphertext ctxt;
    // seal::scheme_type scheme;
    int noise_budget;
    int mult_depth;
    int add_depth;
    double scale;
};

#endif