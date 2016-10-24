#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

// The first thing we do is define our CNN.  The CNN is going to be evaluated
// convolutionally over an entire image pyramid.  Think of it like a normal
// sliding window classifier.  This means you need to define a CNN that can look
// at some part of an image and decide if it is an object of interest.  In this
// example I've defined a CNN with a receptive field of a little over 50x50
// pixels.  This is reasonable for face detection since you can clearly tell if
// a 50x50 image contains a face.  Other applications may benefit from CNNs with
// different architectures.  
// 
// In this example our CNN begins with 3 downsampling layers.  These layers will
// reduce the size of the image by 8x and output a feature map with
// 32 dimensions.  Then we will pass that through 4 more convolutional layers to
// get the final output of the network.  The last layer has only 1 channel and
// the values in that last channel are large when the network thinks it has
// found an object at a particular location.

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


int main(int argc, char** argv)
{
    if (argc != 2)
    {
	cout << "Give the path to the data director as the argument." << endl;
	cout << "For example, ./dnn_mmod face" << endl;
    }

    const std::string data_directory = argv[1];
    
    // Create the variables that will hold our dataset.
    // data_train will hold the images and data_boxes_train will hold
    // the location of the objects in the image
    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> data_boxes_train, data_boxes_test;

    // Load the XML files
    load_image_dataset(images_train, data_boxes_train, data_director+"/training.xml");
    load_image_dataset(images_test, data_boxes_test, data_directory+"/testing.xml");

    cout << "num training images: " << images_train.size() << endl;
    cout << "num testing images: " << images_test.size() << endl;

    // Load MMOD algorithm options
    mmod_options options(data_boxes_train, 40*40);
    cout << "detection window width,height:      " << options.detector_width << "," << options.detector_height << endl;
    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;
 
    // Now we are ready to create our network and trainer
    net_type net(options);
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(5));
    trainer.set_iterations_without_progress_threshold(300);


    // Now we can train the network.  We are going to use mini-batches of 150
    // images.  The images are random crops from our training set
    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels;
    random_cropper cropper;
    cropper.set_chip_dims(200, 200);
    cropper.set_min_object_height(0.2);
    dlib::rand rnd;

    // Run the trainer until the learning rate gets small.
    while( trainer.get_learning_rate() >= 1e-4)
    {
	cropper(150, images_train, data_boxes_train, mini_batch_samples, mini_batch_labels);
	// Randomly change the color to generalize new images better
	for (auto&& img : mini_batch_samples)
	    disturb_colors(img, rnd);

	trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }

    // Wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("cnn_mmod_network.dat") << net;


    // Now that we have a face detector we can test it.  The first statement tests it
    // on the training data.  It will print the precision, recall, and then average precision.
    // This statement should indicate that the network works perfectly on the
    // training data.
    cout << "training results: " << test_object_detection_function(net, images_train, data_boxes_train) << endl;
    // However, to get an idea if it really worked without overfitting we need to run
    // it on images it wasn't trained on.  The next line does this.   Happily,
    // this statement indicates that the detector finds most of the faces in the
    // testing data.
    cout << "testing results:  " << test_object_detection_function(net, images_test, data_boxes_test) << endl;

    // Now lets run the detector on the testing images and look at the outputs.  
    image_window win;
    for (auto&& img : images_test)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
    return 0;
}
