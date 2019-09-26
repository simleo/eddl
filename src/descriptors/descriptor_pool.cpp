//
// Created by Salva Carrión on 26/09/2019.
//

#include "descriptors.h"

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st,
                               const vector<int> &p) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());

    if (ksize.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (pad.size() != 2) msg("Padding must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
}

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st, string p) {
    if (ks.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (st.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

    ksize = ks;
    stride = st;

    if (p == "same") {
        pad.push_back(ksize[0] / 2);
        pad.push_back(ksize[1] / 2);
    } else if (p == "none") {
        pad.push_back(0);
        pad.push_back(0);
    } else msg("Incorrect padding type", "PoolDescriptor::PoolDescriptor");
}


void PoolDescriptor::build(Tensor *A) {
    if (A->ndim != 4) msg("Tensors are not 4D", "PoolDescriptor::build");

    I = A;

    kr = ksize[0];
    kc = ksize[1];

    sr = stride[0];
    sc = stride[1];

    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    padr = pad[0];
    padc = pad[1];

    z = iz;
    r = (ir - kr + 2 * padr) / sr + 1;
    //if (kr%2==0) r--;
    c = (ic - kc + 2 * padc) / sc + 1;
    //if (kc%2==0) c--;

    if ((r <= 0) || (c <= 0))
        msg("Invalid output shape", "PoolDescriptor::build");

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);


}

void PoolDescriptor::resize(Tensor *A) {

    I = A;

    delete O;
    delete D;

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);
}