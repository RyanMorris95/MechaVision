
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <fstream>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

int main( int argc, char** argv)
{
    try
    {
        // Make sure the user entered an argument to this program
        if (argc != 2)
        {
             cout << "error, you have to enter a png file" << endl;
             return 1;
        }

        array2d<rgb_pixel> img;
        load_image(img, argv[1]);
	matrix<rgb_pixel> img2;
	load_image(img2, argv[1]);

	array2d<unsigned int> segment_image;
        array2d<unsigned char> thresh_image;
	const double k = 200;
	const unsigned long min_size = 10;

        auto_threshold_image(img, thresh_image);

	segment_image(img2, segment_image);	
	

	image_window win_segment(segment_image);
        image_window win_thresh(thresh_image);

        win_thresh.wait_until_closed();
    }
    catch (exception& e)
    {
        cout << "exception thrown: " << e.what() << endl;
    }
}
