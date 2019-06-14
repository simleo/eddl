#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "eddl.h"

namespace py = pybind11;

//py::array_t<float> getdata() {
//
//}
//
//void copydata(py::array_t<double> array){}

//Tensor npy2Tensor(py::array_t<double> array){
//    py::buffer_info info = array.request();
//    auto ptr = static_cast<float *>(info.ptr);
//
//    vector<int> s(info.shape.begin(), info.shape.end());
//    Tensor t = Tensor();
//    t.copydata(s, ptr);
//    return t;
//}

// Inner name of the shared library (the python import must much this name and the filename.so)
PYBIND11_MODULE(_C, m) {
    // Constants
    m.attr("DEV_CPU") = DEV_CPU;
    m.attr("DEV_GPU") = DEV_GPU;
    m.attr("DEV_FPGA") = DEV_FPGA;

    // Tensors
    py::class_<Tensor> (m, "Tensor", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<vector<int>, int>())
        .def_readonly("device", &Tensor::device)
        .def_readonly("ndim", &Tensor::ndim)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape);
    //
    //        .def("point2data", &Tensor::point2data)
    //        .def("getdata", &getdata)
    //        .def("copydata", &copydata);

    //m.def("npy2Tensor", &npy2Tensor);

    py::class_<Net>(m, "Model")
        .def("summary", &Net::summary)
        .def("plot", &Net::plot);

    // Optimizer
    py::class_<optim> (m, "Optim");
    // Optimizer: SGD
    py::class_<sgd, optim> (m, "SGD")
        .def(py::init<float, float, float, bool>());

    // Loss
    py::class_<Loss> (m, "Loss");
    // Loss: Cross Entropy
    py::class_<LCrossEntropy, Loss> (m, "LCrossEntropy")
    .def(py::init<>());
    // Loss: Soft Cross Entropy
    py::class_<LSoftCrossEntropy, Loss> (m, "LSoftCrossEntropy")
        .def(py::init<>());
    // Loss: Mean Squared Error
    py::class_<LMeanSquaredError, Loss> (m, "LMeanSquaredError")
    .def(py::init<>());

    // Metric
    py::class_<Metric> (m, "Metric");
    // Metric: Categorical Accuracy
    py::class_<MCategoricalAccuracy, Metric> (m, "MCategoricalAccuracy")
        .def(py::init<>());
    // Metric: Mean Squared Error
    py::class_<MMeanSquaredError, Metric> (m, "MMeanSquaredError")
    .def(py::init<>());

    // Computing service
    py::class_<CompServ>(m, "CompServ");

    // EDDL
    py::class_<EDDL>(m, "EDDL")
        .def(py::init<>())
        .def("CS_CPU", &EDDL::CS_CPU)
        .def("build", &EDDL::build2)
        .def("get_model_mlp", &EDDL::get_model_mlp)
        .def("get_model_cnn", &EDDL::get_model_cnn);
}