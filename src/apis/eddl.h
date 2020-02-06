/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _EDDL_
#define _EDDL_

#include <vector>
#include <thread>
#include <pthread.h>
#include <functional>

#include "../net/net.h"
#include "../net/netloss.h"
#include "../initializers/initializer.h"
#include "../regularizers/regularizer.h"
#include "../losses/loss.h"
#include "../metrics/metric.h"

#include "../layers/layer.h"
#include "../layers/conv/layer_conv.h"
#include "../layers/core/layer_core.h"
#include "../layers/da/layer_da.h"
#include "../layers/generators/layer_generators.h"
#include "../layers/merge/layer_merge.h"
#include "../layers/noise/layer_noise.h"
#include "../layers/normalization/layer_normalization.h"
#include "../layers/operators/layer_operators.h"
#include "../layers/reductions/layer_reductions.h"
#include "../layers/pool/layer_pool.h"
#include "../layers/recurrent/layer_recurrent.h"


// EDDL namespace defines the API
namespace eddl {

typedef Layer* layer;
typedef Net* model;
typedef Optimizer* optimizer;
typedef Initializer* initializer;
typedef Regularizer* regularizer;
typedef CompServ* compserv;
typedef NetLoss * loss;
typedef NetLoss * metric;

    ///////////////////////////////////////
    //  MODEL METHODS
    ///////////////////////////////////////

    // Creation
    /**
      *  @brief Model instance.
      *
      *  @param in  Vector with model input layers, typically Input({...})
      *  @param out  Vector with the ouputs of the model. Example: {Sigmoid(MyModel())}
      *  @return     Model instance
    */
    model Model(vlayer in, vlayer out);
    void build(model net, optimizer o=nullptr, CompServ *cs=nullptr, bool init_weigths=true);
    /**
      *  @brief Tell the model which optimizer, losses, metrics and computing services to use. Losses and metrics are specified by name.
      *
      *  @param net  Model
      *  @param o  Optimizer
      *  @param lo  Vector with loss names
      *  @param me  Vector with metric names
      *  @param cs  Computing service
      *  @return     (void)
    */
    void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs=nullptr, bool init_weights=true);

    /**
      *  @brief Tell the model which optimizer, losses, metrics and computing services to use.
      *
      *  @param net  Model
      *  @param o  Optimizer
      *  @param lo  Vector with losses
      *  @param me  Vector with metrics
      *  @param cs  Computing service
      *  @return     (void)
    */
    void build(model net, optimizer o, const vector<Loss*> &lo, const vector<Metric*> &me, CompServ *cs=nullptr, bool init_weights=true);

    // Computing services
    /**
      *  @brief Assign model operations to the GPU.
      *
      *  @param net  Model
      *  @param g  Vector with gpu ids to allocate the model
      *  @param lsb  Number of batches to sync model weights
      *  @return     (void)
    */
    void toGPU(model net, vector<int> g,int lsb);
    void toGPU(model net, vector<int> g,string mem);
    void toGPU(model net, vector<int> g,int lsb, string mem);
    void toGPU(model net, vector<int> g);
    void toGPU(model net, string mem);
    void toGPU(model net);
    //void toGPU(model net, string mem);
    /**
      *  @brief Assign model operations to the CPU.
      *
      *  @param net  Model
      *  @param t  CPU Threads
      *  @return     (void)
    */
    void toCPU(model net, int t=std::thread::hardware_concurrency());
    compserv CS_CPU(int th=-1);

    compserv CS_GPU();
    compserv CS_GPU(const vector<int> g);
    compserv CS_GPU(const vector<int> g,int lsb);
    compserv CS_GPU(const vector<int> g,string mem);
    compserv CS_GPU(const vector<int> g,int lsb,string mem);

    compserv CS_FGPA(const vector<int> &f,int lsb=1);
    compserv CS_COMPSS(string filename);


    // Info and logs
    void setlogfile(model net,string fname);
    /**
      *  @brief  Prints a summary representation of your model.
      *
      *  @param m  Model to train
      *  @return     (void) Prints the model
    */
    void summary(model m);
    /**
      *  @brief  Plots a representation of your model.
      *
      *  @param m  Model to plot
      *  @param fname  Where the plot is saved
      *  @return     (void) Plots the model
    */
    void plot(model m, string fname, string mode="LR");

    // Serialization
    /**
      *  @brief  Load weights to reinstantiate your model.
      *
      *  @param m  Model
      *  @param fname  Where are the model weights
      *  @return     (void) Load the weights
    */
    void load(model m, const string& fname, string format="bin");
    /**
      *  @brief  Save weights of a model.
      *
      *  @param m  Model
      *  @param fname  Where the model weights will be saved
      *  @return     (void) Save the weights
    */
    void save(model m, const string& fname, string format="bin");

    // Optimizer
    /**
      *  @brief  Changes the learning rate and hyperparameters of the model optimizer.
      *
      *  @param net  Model
      *  @param p  Vector with the learning rate and hyperparameters of the model
      *  @return     (void) Changes model optimizer settings
    */
    void setlr(model net,vector<float>p);
    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement
    /**
      *  @brief Adam optimizer.
      *
      *  @see   https://arxiv.org/abs/1412.6980v8
      *
      *  @param lr  Learning rate
      *  @param beta_1  Coefficients used for computing running averages of gradient and its square
      *  @param beta_2  Coefficients used for computing running averages of gradient and its square
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @param amsgrad   Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".
      *  @return     Adam optimizer
    */
    optimizer adam(float lr=0.01, float beta_1=0.9, float beta_2=0.999, float epsilon=0.000001, float weight_decay=0,bool amsgrad=false); //Todo: Implement
    optimizer adagrad(float lr, float epsilon, float weight_decay); //Todo: Implement
    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay); //Todo: Implement
    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay); //Todo: Implement
    /**
      *  @brief RMSProp optimizer.
      *
      *  @details
      *   It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned).
      *
      *   @see  http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
      *
      *  @param lr  Learning rate
      *  @param rho  Smoothing constant
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @return     RMSProp optimizer
    */
    optimizer rmsprop(float lr=0.01, float rho=0.9, float epsilon=0.00001, float weight_decay=0.0); //Todo: Implement
    /**
      *  @brief Stochastic gradient descent optimizer.
      *
      *  @details
      *   Includes support for momentum, learning rate decay, and Nesterov momentum
      *
      *  @param lr  Learning rate
      *  @param momentum  Momentum factor
      *  @param weight_decay   Value to apply to the activation function
      *  @param nesterov   Boolean. Whether to apply Nesterov momentum
      *  @return     Stochastic gradient descent optimizer
    */
    optimizer sgd(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

    // Training and Evaluation
    // Coarse methods
    /**
      *  @brief Trains the model for a fixed number of epochs (iterations on a dataset).
      *
      *  @param m  Model to train
      *  @param in  Input data (features)
      *  @param out  Output data (labels)
      *  @param batch  Number of samples per gradient update
      *  @param epochs  Number of epochs to train the model. An epoch is an iteration over the entire data provided
      *  @return     (void) Trains the model
    */
    void fit(model m, const vector<Tensor *> &in, const vector<Tensor *> &out, int batch, int epochs);
    /**
      *  @brief Returns the loss value & metrics values for the model in test mode.
      *
      *  @param m  Model to train
      *  @param in  Input data (features)
      *  @param out  Output data (labels)
      *  @return     (void) Evaluates the model
    */
    void evaluate(model m, const vector<Tensor *> &in, const vector<Tensor *> &out);

    // Finer methods
    vector<int> random_indices(int batch_size, int num_samples);
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void next_batch(vector<Tensor *> in,vector<Tensor *> out);
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out);
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out);

    // Finest methods
    /**
      *  @brief Set model mode.
      *
      *  @param net  Model
      *  @param mode  Train 1, Test 0
      *  @return     (void)
    */
    void set_mode(model net, int mode);
    /**
      *  @brief Resets model loss.
      *
      *  @param net  Model
      *  @return     (void)
    */
    void reset_loss(model m);
    vlayer forward(model m,vector<Layer *> in);
    vlayer forward(model m,vector<Tensor *> in);
    vlayer forward(model m);
    vlayer forward(model m,int b);
    /**
      *  @brief Set model gradients to zero.
      *
      *  @param net  Model
      *  @return     (void)
    */
    void zeroGrads(model m);
    /**
      *  @brief Calculates the gradient by passing it's argument (1x1 unit tensor by default) through the backward graph.
      *
      *  @param net  Model
      *  @param target  Targets
      *  @return     (void)
    */
    void backward(model m,vector<Tensor *> target);
    void backward(model net);
    void backward(loss l);
    void update(model m);
    /**
      *  @brief Prints model loss at some batch.
      *
      *  @param net  Model
      *  @param batch  Batch number
      *  @return     (void)
    */
    void print_loss(model m, int batch);

    // model constraints
    /**
      *  @brief Model parameters values clipping.
      *
      *  @param m  Model
      *  @param min  Minimum value
      *  @param max   Maximum value
      *  @return     (void) Performs model clamp between min and max
    */
    void clamp(model m,float min,float max);

    // loss and metrics methods
    float compute_loss(loss L);
    float compute_metric(loss L);
    /**
      *  @brief Get Loss by his name.
      *
      *  @param type  Loss name/type
      *  @return     Selected Loss
    */
    Loss* getLoss(string type);
    /**
      *  @brief Create new Loss.
      *
      *  @param f  Loss function
      *  @param in  Loss input
      *  @param name  Loss name
      *  @return     Created Loss
    */
    loss newloss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    /**
      *  @brief Create new Loss.
      *
      *  @param f  Loss function
      *  @param in  Loss input
      *  @param name  Loss name
      *  @return     Created Loss
    */
    loss newloss(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
    /**
      *  @brief Get Metric by his name.
      *
      *  @param type  Metric name/type
      *  @return     Selected Metric
    */
    Metric* getMetric(string type);
    loss newmetric(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    loss newmetric(const std::function<Layer*(Layer*)>& f, Layer *in, string name);

    // graph connections
    layer detach(layer l);
    vlayer detach(vlayer l);


    ///////////////////////////////////////
    //  LAYERS
    ///////////////////////////////////////

    // Core Layers
    /**
      *  @brief Solves non-linear equation with Newton method.
      *
      *  @details
      *   Applies an activation function to the given layer
      *
      *  @see   https://en.wikipedia.org/wiki/Activation_function
      *
      *  @param parent  Parent layer
      *  @param activation Name of the activation function
      *  @param param   Value to apply to the activation function
      *  @param name  Name of the layer
      *  @return     Activation layer
    */
    layer Activation(layer parent, string activation, float param=0.01, string name = "");

    /**
      *  @brief Applies a Softmax activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Softmax_function
      *
      *  @param parent  Parent layer
      *  @return     Output of Softmax transformation
    */
    layer Softmax(layer parent);

    /**
      *  @brief Applies a Sigmoid activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Sigmoid_function
      *
      *  @param parent  Parent layer
      *  @return     Output of Sigmoid activation
    */
    layer Sigmoid(layer parent);

    /**
      *  @brief Applies a Rectified Linear Unit activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
      *
      *  @param parent  Parent layer
      *  @return     Output of ReLu activation
    */
    layer ReLu(layer parent);

    /**
      *  @brief Applies the Leaky version of a Rectified Linear Unit activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs
      *
      *  @param parent  Parent layer
      *  @param param  Negative slope coefficient
      *  @return     Output of Leaky ReLu activation
    */
    layer LeakyReLu(layer parent, float param=0.01);

    /**
      *  @brief Applies the Exponential Linear Unit activation function to the given layer.
      *
      *  @param parent  Parent layer
	  *  @param param ELu coefficient
      *  @return     Output of ELu activation
    */
    layer Elu(layer parent, float param=1.0);

    /**
      *  @brief Applies the Scaled Exponential Linear Unit activation function to the given layer.
      *
      *  @param parent  Parent layer
      *  @return     Output of Selu activation
    */
    layer Selu(layer parent);

    /**
    *  @brief Applies the Exponential (base e) activation function to the given layer.
    *
    *  @param parent  Parent layer
    *  @return     Output of Exponential activation
    */
    layer Exponential(layer parent);

    /**
    *  @brief Applies the Softplus activation function to the given layer.
    *
    *  @param parent  Parent layer
    *  @return     Output of Exponential activation
    */
    layer Softplus(layer parent);


    /**
    *  @brief Applies the Softsign activation function to the given layer.
    *
    *  @param parent  Parent layer
    *  @return     Output of Exponential activation
    */
    layer Softsign(layer parent);

    /**
      *  @brief Applies the Linear activation function to the given layer.
      *
      *  @param parent  Parent layer
	  *  @param param Linear coefficient
      *  @return     Output of Linear activation
    */
    layer Linear(layer parent, float param=1.0);

    /**
      *  @brief Applies the Hyperbolic tangent activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Hyperbolic_function
      *
      *  @param parent  Parent layer
      *  @return     Output of hyperbolic activation
    */
    layer Tanh(layer parent);

    /**
      *  @brief Convolution layer.
      *
      *  @param parent  Parent layer
      *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
      *  @param kernel_size  Vector of 2 integers, specifying the height and width of the 2D convolution window.
      *  @param strides  Vector of 2 integers, specifying the strides of the convolution along the height and width
      *  @param padding  One of "none", "valid" or "same"
      *  @param groups  Number of blocked connections from input channels to output channels
      *  @param dilation_rate  Vector of 2 integers, specifying the dilation rate to use for dilated convolution
      *  @param use_bias  Boolean, whether the layer uses a bias vector.
      *  @param name  A name for the operation
      *  @return     Convolution layer
    */
    layer Conv(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides = {1, 1}, string padding = "same", int groups = 1,
               const vector<int> &dilation_rate = {1, 1},
               bool use_bias = true, string name = "");

    /**
      *  @brief Regular densely-connected NN layer.
      *
      *  @param parent  Parent layer
      *  @param ndim  Positive integer, dimensionality of the output space
      *  @param use_bias  Boolean, whether the layer uses a bias vector.
      *  @param name  A name for the operation
      *  @return     Densely-connected NN layer
    */
    layer Dense(layer parent, int ndim, bool use_bias = true,  string name = "");

    /**
      *  @brief Applies Dropout to a layer.
      *
      *  @details
      *   Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
      *
      *  @param parent  Parent layer
      *  @param rate  Between 0 and 1. Fraction of the input units to drop
      *  @param name  A name for the operation
      *  @return     Layer with Dropout
    */
    layer Dropout(layer parent, float rate, string name = "");

    /**
      *  @brief Used to initialize an input to a model.
      *
      *  @param shape  A shape vector (integer), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors
      *  @param name  A name for the operation
      *  @return     Input layer
    */
    layer Input(const vector<int> &shape, string name = "");

    /**
      *  @brief Upsampling layer.
      *
      *  @details
      *   Repeats the rows and columns of the data.
      *
      *  @param parent  Parent layer
      *  @param size  Vector of 2 integers. The upsampling factors for rows and columns
      *  @param interpolation  A string, one of nearest or bilinear
      *  @param name  A name for the operation
      *  @return     Output layer after upsampling operation
    */
    layer UpSampling(layer parent, const vector<int> &size, string interpolation = "nearest", string name = "");

    /**
      *  @brief Reshapes an output to a certain shape.
      *
      *  @param parent  Parent layer
      *  @param shape  Target shape. Vector of integers. Does not include the batch axis
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Reshape(layer parent, const vector<int> &shape, string name = "");

    /**
      *  @brief Flattens the input. Does not affect the batch size.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Flatten(layer parent, string name = "");

    layer ConvT(layer parent, int filters, const vector<int> &kernel_size,
                const vector<int> &output_padding, string padding = "same",
                const vector<int> &dilation_rate = {1, 1},
                const vector<int> &strides = {1, 1}, bool use_bias = true, string name = ""); //Todo: Implement
    layer Embedding(int input_dim, int output_dim, string name = ""); //Todo: Implement

    /**
      *  @brief Transposes a Layer.
      *
      *  @param parent  Parent layer
      *  @param dims  Vector of integers with the transpose dimensions
      *  @param name  A name for the operation
      *  @return     Output of transpose operation
    */
    layer Transpose(layer parent, string name = "");

    // Transformation Layers
    /**
      *  @brief Affine transformation of the image keeping center invariant: rotate+translate+scale+shear.
      *
      *  @param parent  Parent layer
      *  @param angle  Angle factor
      *  @param translate  Translate factor
      *  @param scale  Scaling factor
      *  @param shear  Shear factor
      *  @param name  A name for the operation
      *  @return     Output of affine transformation
    */
    layer Affine(layer parent, float angle=0, float translate=0, float scale=0, float shear=0, string name="");  // TODO: Implement
    /**
      *  @brief Crops the given image at `[(top, left), (bottom, right)]`.
      *
      *  @param parent  Parent layer
      *  @param from_coords  Vector (top, left) coordinates
      *  @param to_coords  Vector (bottom, right) coordinates
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of crop transformation
    */
    layer Crop(layer parent, vector<int> from_coords, vector<int> to_coords, bool reshape=true, float constant=0.0f, string name="");
    /**
      *  @brief Crops the given image at the center with size (width, height).
      *
      *  @param parent  Parent layer
      *  @param size  Vector (height, width) size
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of center crop transformation
    */
    layer CenteredCrop(layer parent, vector<int> size, bool reshape=true, float constant=0.0f, string name="");
    /**
      *  @brief Randomly change the brightness, contrast and saturation of an image.
      *
      *  @param parent  Parent layer
      *  @param brightness  How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers
      *  @param contrast  How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers
      *  @param saturation  How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers
      *  @param hue  How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5
      *  @param name  A name for the operation
      *  @return     Output of color jitter transformation
    */
    layer ColorJitter(layer parent, float brightness=0, float contrast=0, float saturation=0, float hue=0, string name="");  // TODO: Implement
    /**
      *  @brief Crop the given image at `[(top, left), (bottom, right)]` and scale it to the parent size.
      *
      *  @param parent  Parent layer
      *  @param from_coords  Vector (top, left) coordinates
      *  @param to_coords  Vector (bottom, right) coordinates
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirrror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the rotated image, it is used for all channels respectively
      *  @param name  A name for the operation
      *  @return     Output of crop scale transformation
    */
    layer CropScale(layer parent, vector<int> from_coords, vector<int> to_coords, string da_mode="nearest", float constant=0.0f, string name="");
    /**
      *  @brief Selects a rectangle region in an image at `[(top, left), (bottom, right)]` and erases its pixels using a constant value.
      *
      *  @param parent  Parent layer
      *  @param from_coords  Vector (top, left) coordinates
      *  @param to_coords  Vector (bottom, right) coordinates
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of cutout transformation
    */
    layer Cutout(layer parent, vector<int> from_coords, vector<int> to_coords, float constant=0.0f, string name="");
    /**
      *  @brief Flip the given image at `axis=n`.
      *
      *  @param parent  Parent layer
      *  @param axis  Flip axis
      *  @param name  A name for the operation
      *  @return     Output of flip transformation
    */
    layer Flip(layer parent, int axis=0, string name="");
    layer Grayscale(layer parent,  string name="");  // TODO: Implement
    /**
      *  @brief Horizontally flip the given image.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of horizontal flip transformation
    */
    layer HorizontalFlip(layer parent, string name="");
    layer Pad(layer parent, vector<int> padding, float constant=0.0f, string name=""); // TODO: Implement
    /**
      *  @brief Rotate the image by angle.
      *
      *  @param parent  Parent layer
      *  @param angle  In degrees counter clockwise order
      *  @param offset_center  Optional center of rotation. Default is the center of the image
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirrror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the rotated image, it is used for all channels respectively.
      *  @return     Output of rotate transformation
    */
    layer Rotate(layer parent, float angle, vector<int> offset_center={0, 0}, string da_mode="original", float constant=0.0f, string name="");
    /**
      *  @brief Resize the input image to the given size. `[height, width]`.
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector with layer/images desired new shape
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirrror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively.
      *  @return     Output of scale transformation
    */
    layer Scale(layer parent, vector<int> new_shape, bool reshape=true, string da_mode="nearest", float constant=0.0f, string name="");
    /**
      *  @brief Shift the input image `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param shift  Vector of maximum absolute fraction for horizontal and vertical translations
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirrror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively.
      *  @return     Output of scale transformation
    */
    layer Shift(layer parent, vector<int> shift, string da_mode="nearest", float constant=0.0f, string name="");
    /**
      *  @brief Vertically flip the given image.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of vertical flip transformation
    */
    layer VerticalFlip(layer parent, string name="");
    /**
      *  @brief Normalize an image with mean and standard deviation.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of normalize transformation
    */
    layer Normalize(layer parent, string name="");  // TODO: Implement


    // Data augmentation Layers
    /**
      *  @brief Random affine transformation of the image keeping center invariant: rotate+translate+scale+shear.
      *
      *  @param parent  Parent layer
      *  @param angle  Angle factor range
      *  @param translate  Translate factor range
      *  @param scale  Scaling factor range
      *  @param shear  Shear factor range
      *  @param name  A name for the operation
      *  @return     Output of affine transformation
    */
    layer RandomAffine(layer parent, vector<float> angle, vector<float> translate, vector<float> scale, vector<float> shear, string name="");  // TODO: Implement
    /**
      *  @brief Crop the given image at a random location with size `[height, width]`.
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector (height, width) size
      *  @param name  A name for the operation
      *  @return     Output of random crop transformation
    */
    layer RandomCrop(layer parent, vector<int> new_shape, string name= "");
    /**
      *  @brief Crops the given image at the center with size (width, height).
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector (height, width) size
      *  @param name  A name for the operation
      *  @return     Output of random center crop transformation
    */
    layer RandomCenteredCrop(layer parent, vector<int> new_shape, string name= "");  // TODO: Implement
    /**
      *  @brief Crop the given image randomly by the size in a range `[a, b]` by and scale it to the parent size.
      *
      *  @param parent  Parent layer
      *  @param factor  Factor Range for random crop
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirrror", "reflect", "wrap", "original")
      *  @param name  A name for the operation
      *  @return     Output of random crop scale transformation
    */
    layer RandomCropScale(layer parent, vector<float> factor, string da_mode= "nearest", string name= "");
    /**
      *  @brief Randomly selects a rectangle region in an image and erases its pixels. The random region is defined by the range `[(min_x, max_x), (min_y, max_y)]`, where these are relative values.
      *
      *  @param parent  Parent layer
      *  @param factor_x  Vector of factor fraction for horizontal size
      *  @param factor_y  Vector of factor fraction for vertical size
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of random cutout transformation
    */
    layer RandomCutout(layer parent, vector<float> factor_x, vector<float> factor_y, float constant= 0.0f, string name= "");
    /**
      *  @brief Flip the given image at `axis=n` randomly with a given probability.
      *
      *  @param parent  Parent layer
      *  @param axis  Flip axis
      *  @param name  A name for the operation
      *  @return     Output of random flip transformation
    */
    layer RandomFlip(layer parent, int axis, string name= "");
    layer RandomGrayscale(layer parent, string name= "");
    /**
      *  @brief Horizontally flip the given image randomly with a given probability.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of random horizontal flip transformation
    */
    layer RandomHorizontalFlip(layer parent, string name= "");
    /**
      *  @brief Rotate the image randomly by an angle defined in a range `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param factor  Range In degrees counter clockwise order
      *  @param offset_center  Optional center of rotation. Default is the center of the image
      *  @param da_mode  One of "original"
      *  @param constant  Fill value for area outside the rotated image, it is used for all channels respectively.
      *  @return     Output of rotate transformation
    */
    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center= {0, 0}, string da_mode= "original", float constant= 0.0f, string name= "");
    /**
      *  @brief Resize the input image randomly by the size in a range `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param factor  Vector of factor size range new shape
      *  @param da_mode  One of "nearest"
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively.
      *  @return     Output of scale transformation
    */
    layer RandomScale(layer parent, vector<float> factor, string da_mode= "nearest", float constant= 0.0f, string name= "");
    /**
      *  @brief Shift the input image randomly in range `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param factor_x  Vector of factor fraction for horizontal translations
      *  @param factor_y  Vector of factor fraction for vertical translations
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirrror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively.
      *  @return     Output of scale transformation
    */
    layer RandomShift(layer parent, vector<float> factor_x, vector<float> factor_y, string da_mode= "nearest", float constant= 0.0f, string name= "");
    /**
      *  @brief Vertically flip the given image randomly with a given probability.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of random vertical flip transformation
    */
    layer RandomVerticalFlip(layer parent, string name= "");

    // Merge Layers
    /**
      *  @brief Layer that adds a list of layer inputs.
      *
      *  @details
      *   It takes as input a list of layers, all of the same shape, and returns a single tensor (also of the same shape).
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of add operation with all input layers
    */
    layer Add(const vector<layer> &layers, string name = "");
    /**
      *  @brief Layer that averages a list of layer inputs.
      *
      *  @details
      *   It takes as input a list of layers, all of the same shape, and returns a single tensor (also of the same shape).
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of average operation with all input layers
    */
    layer Average(const vector<layer> &layers, string name = ""); //Todo: Implement

    /**
      *  @brief Layer that concatenates a list of inputs.
      *
      *  @details
      *   It takes as input a list of layers and returns a single tensor, the concatenation of all inputs.
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of concatenation operation with all input layers
    */
    layer Concat(const vector<layer> &layers, unsigned int axis=1, string name = "");
    layer MatMul(const vector<layer> &layers, string name = "");
    layer Maximum(const vector<layer> &layers, string name = "");
    layer Minimum(const vector<layer> &layers, string name = "");
    layer Subtract(const vector<layer> &layers, string name = "");


    // Noise Layers
    /**
      *  @brief Apply additive zero-centered Gaussian noise.
      *
      *  @details
      *   This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
      *   Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.
      *   As it is a regularization layer, it is only active at training time.
      *
      *  @param parent  Parent layer
      *  @param stddev  Standard deviation of the noise distribution
      *  @param name  A name for the operation
      *  @return     The parent after apply the GaussianNoise layer
    */
    layer GaussianNoise(layer parent, float stddev, string name = "");

    // Normalization
    /**
      *  @brief Batch normalization layer.
      *
      *  @details
      *   Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
      *
      *  @see   https://arxiv.org/abs/1502.03167
      *
      *  @param parent  Parent layer
      *  @param momentum  Momentum for the moving mean and the moving variance
      *  @param epsilon  Small float added to variance to avoid dividing by zero
      *  @param affine  A boolean value that when set to True, this module has learnable affine parameters
      *  @param name  A name for the operation
      *  @return     Parent layer after the normalization
    */
    layer BatchNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");

    /**
      *  @brief Layer normalization layer.
      *
      *  @details
      *   Applies Layer Normalization over a input.
      *
      *  @see   https://arxiv.org/abs/1607.06450
      *
      *  @param parent  Parent layer
      *  @param momentum  Momentum for the moving mean and the moving variance
      *  @param epsilon  Value added to the denominator for numerical stability
      *  @param affine  A boolean value that when set to True, this module has learnable affine parameters
      *  @param name  A name for the operation
      *  @return     Parent layer after the normalization
    */
    layer LayerNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");

    /**
      *  @brief Group normalization layer.
      *
      *  @details
      *   Divides the channels into groups and computes within each group the mean and variance for normalization. The computation is independent of batch sizes.
      *
      *  @see   https://arxiv.org/abs/1803.08494
      *
      *  @param parent  Parent layer
      *  @param groups  Number of groups in which the channels will be divided
      *  @param momentum  Momentum for the moving mean and the moving variance
      *  @param epsilon  Value added to the denominator for numerical stability
      *  @param affine  A boolean value that when set to True, this module has learnable affine parameters
      *  @param name  A name for the operation
      *  @return     Parent layer after the normalization
    */
    layer GroupNormalization(layer parent, int groups, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");
    layer Norm(layer parent, float epsilon = 0.001f, string name = "");
    layer NormMax(layer parent, float epsilon = 0.001f, string name = "");
    layer NormMinMax(layer parent, float epsilon = 0.001f, string name = "");


    //  Operator Layers
    layer Abs(layer l);
    layer Diff(layer l1, layer l2);
    layer Diff(layer l1, float k);
    /**
      *  @brief Layer that computes the difference of a float number and a layer.
      *
      *  @param k  Number
      *  @param l1  Parent layer
      *  @return     Parent layer l1 after computing his difference with k
    */
    layer Diff(float k, layer l1);
    layer Div(layer l1, layer l2);
    layer Div(layer l1, float k);
    layer Div(float k, layer l1);
    layer Exp(layer l);
    /**
      *  @brief Layer that computes the logarithm of a layer.
      *
      *  @param l  Parent layer
      *  @return     Parent layer l after computing his logarithm
    */
    layer Log(layer l);
    layer Log2(layer l);
    layer Log10(layer l);
    layer Mult(layer l1, layer l2);
    /**
      *  @brief Layer that computes the multiplication of a float number and a layer.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer l1 after computing his multiplication with k
    */
    layer Mult(layer l1, float k);
    layer Mult(float k,layer l1);
    layer Pow(layer l1, layer l2);
    layer Pow(layer l1, float k);
    layer Sqrt(layer l);
    /**
      *  @brief Layer that computes the sum of two layers.
      *
      *  @param l1  Layer
      *  @param l2  Layer
      *  @return     The result after computing the sum between layers l1 and l2
    */
    layer Sum(layer l1, layer l2);
    /**
      *  @brief Layer that computes the sum of a float number and a layer.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer l1 after computing his sum with k
    */
    layer Sum(layer l1, float k);
    layer Sum(float k, layer l1);
    layer Select(layer l, vector<string> indices, string name="");
    layer Permute(layer l, vector<int> dims, string name="");

    // Reduction Layers
    layer ReduceMean(layer l, vector<int> axis = {0}, bool keepdims = false);
    layer ReduceVar(layer l, vector<int> axis = {0}, bool keepdims = false);
    layer ReduceSum(layer l, vector<int> axis = {0}, bool keepdims = false);
    layer ReduceMax(layer l, vector<int> axis = {0}, bool keepdims = false);
    layer ReduceMin(layer l, vector<int> axis = {0}, bool keepdims = false);

    // Generator Layers
    layer GaussGenerator(float mean, float stdev, vector<int> size);
    layer UniformGenerator(float low, float high, vector<int> size);

    // Pooling Layers
    /**
      *  @brief Average pooling operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the average pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive).
      *  @param name  A name for the operation
      *  @return     The result after apply the average pooling operation over the parent layer
    */
    layer AveragePool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},string padding = "none", string name = "");
    /**
      *  @brief Global Max pooling operation.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     The result after apply the global max pooling operation over the parent layer
    */
    layer GlobalMaxPool(layer parent, string name = ""); //Todo: Implement
    /**
      *  @brief Global Average pooling operation.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     The result after apply the global average pooling operation over the parent layer
    */
    layer GlobalAveragePool(layer parent, string name = ""); //Todo: Implement
    /**
      *  @brief Max pooling operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the max pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive).
      *  @param name  A name for the operation
      *  @return     The result after apply the max pooling operation over the parent layer
    */
    layer MaxPool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2}, string padding = "none", string name = "");

    // Recurrent Layers
    layer RNN(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");
    layer LSTM(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");


    // Layers Methods
    void set_trainable(layer l, bool val);
    void copyTensor(Layer *l1,Layer *l2);
    void copyGrad(Layer *l1,Layer *l2);
    vlayer getOut(model net);
    Tensor* getTensor(layer l);
    Tensor* getGrad(layer l);


    ///////////////////////////////////////
    //  INITIALIZERS
    ///////////////////////////////////////
    /**
      *  @brief Glorot normal initializer, also called Xavier normal initializer.
      *
      *  @details
      *   It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
      *
      *  @param l  Parent layer to initialize
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with the Glorot normal
    */
    layer GlorotNormal(layer l,int seed=1234);
    /**
      *  @brief Glorot uniform initializer, also called Xavier uniform initializer.
      *
      *  @details
      *   It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
      *
      *  @param l  Parent layer to initialize
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with the Glorot uniform
    */
    layer GlorotUniform(layer l,int seed=1234);
    /**
      *  @brief Random normal initializer.
      *
      *  @param l  Parent layer to initialize
      *  @param m  Mean of the normal distribution to draw samples
      *  @param s  Standard deviation of the normal distribution to draw samples
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with a random normal distribution
    */
    layer RandomNormal(layer l, float m=0.0,float s=0.1, float seed=1234);
    layer RandomUniform(layer l, float min=0.0,float max=0.1, float seed=1234);
    /**
      *  @brief Initializer that generates tensors initialized to a constant value.
      *
      *  @param l  Parent layer to initialize
      *  @param v   Value of the generator
      *  @return     The layer l initialized with a constant value
    */
    layer Constant(layer l, float v=0.1);


    ///////////////////////////////////////
    //  REGULARIZERS
    ///////////////////////////////////////
    layer L2(layer l,float l2);
    layer L1(layer l,float l1);
    layer L1L2(layer l,float l1,float l2);


    ///////////////////////////////////////
    //  DATASETS
    ///////////////////////////////////////
    bool exist(string name);
    /**
      *  @brief Downloads MNIST Dataset.
      *
      *  @see   http://yann.lecun.com/exdb/mnist/
      *
      *  @return     (void) The binary files of MNIST
    */
    void download_mnist();
    /**
      *  @brief Downloads CIFAR-10 Dataset.
      *
      *  @see   https://www.cs.toronto.edu/~kriz/cifar.html
      *
      *  @return     (void) The binary files of CIFAR-10
    */
    void download_cifar10();
    /**
      *  @brief Downloads DRIVE Dataset.
      *
      *  @see   https://drive.grand-challenge.org/
      *
      *  @return     (void) The numpy files of DRIVE
    */
    void download_drive();

}
#endif
