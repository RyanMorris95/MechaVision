
/*
    Using opencv to capture camera footage and then runs the 
    footage through the neural network to detect objects. 
    Once an object is detected the we track the object.
*/


// Try compiling with cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF . if you get an error -lopencv_dep_cudart

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/gui_widgets.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/geometry.h>
//#include <opencv2/imgcodecs/imgcodecs.hpp>

/*
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include <NVXIO/Application.hpp>
#include <NVXIO/ConfigParser.hpp>
#include <NVXIO/FrameSource.hpp>
#include <NVXIO/Render.hpp>
#include <NVXIO/SyncTimer.hpp>
#include <NVXIO/Utility.hpp>

#include "stereo_matching.hpp"
*/
using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------
// Let's begin the network definition by creating some network blocks.

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon5  = relu<bn_con<con5<45,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type  = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;



// ---------------------- Function Prototypes --------------------------------------------
//static void displayState( nvxio::Render *renderer, const nvxio::FrameSource::Parameters &sourceParams, double proc_ms, double total_ms );
//static vool read( const std::string &nf, StereoMatching::StereoMatchingParams &config, std::string &message);
//static void eventCallback(void *eventData, vx_char key, vx_uint32, vx_uint32);

// --------------------- Global Types ---------------------------------------------------
/*
enum OUTPUT_IMAGE
{
    ORIG_FRAME,
    ORIG_DISPARITY,
    COLOR_OUTPUT
};
*/


// ------------------- Main Function ----------------------------------------------------
int main(int argc, char** argv) try
{
    
    cv::VideoCapture cap("IMG_2699.MOV");
    //cv::VideoCapture cap(0); 
    //cv::VideoCapture cap(1);
    if (!cap.isOpened())
    {
	cerr << "Unable to connect to camera" << endl;
	return 1;
    }

    net_type net;
    deserialize(argv[1]) >> net;  

    image_window win;
    image_window seg_win;
    
    // Create a correlation tracker and then start in on the first detection
    correlation_tracker tracker;
    int num_dets = 0;
    int frame_count = 0;

    // Grab and process frames until the main window is closed by the user.
    while(!win.is_closed())
    {
	// Grab a frame
	cv::Mat temp;
	cap >> temp;
	cv::cvtColor(temp, temp, CV_BGR2RGB);


	// Turn OpenCV's Mat into something dlib can deal with.
	// Don't modify temp while using cimg
	cv_image<rgb_pixel> cimg(temp);


	// Downscale image
        matrix<rgb_pixel> out_img;
	matrix<rgb_pixel> roi_img;
        out_img.set_size(400,400);

        resize_image(cimg, out_img);

        // Upsampling the image will allow us to detect smaller faces but will cause the
        // program to use more RAM and run longer.
        //while(img.size() > 100*100)
            //pyramid_down(img);

        // Note that you can process a bunch of images in a std::vector at once and it runs
        // much faster, since this will form mini-batches of images and therefore get
        // better parallelism out of your GPU hardware.  However, all the images must be
        // the same size.  To avoid this requirement on images being the same size we
        // process them individually in this example.
	win.clear_overlay();
	win.set_image(out_img);
		
	if (frame_count == 10)
	{
		cout << "Should be detecting." << endl;
	        auto dets = net(out_img);	
		// Overlay detection on window
		for (auto&& d : dets)
		{
		     cout << "Detections at: ";
		     cout << d;

		     win.add_overlay(d);
		     // Start track on first detection
		     
		     tracker.start_track(out_img, centered_rect(point(center(d.rect)), d.rect.width(), d.rect.height()));
		     //extract_image_chips(out_img, d , roi_img);
		     segment_image(out_img, roi_img, 200, 10);
		     num_dets++;
		     
		}
		frame_count = 0;
	}
	// Now update tracker each frame
	if (num_dets > 0)
	{
	    tracker.update(out_img);
	    win.add_overlay(tracker.get_position());
            
	}

	frame_count++;
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

