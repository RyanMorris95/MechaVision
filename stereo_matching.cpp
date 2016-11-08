#include "stereo_matching.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>

#include <NVXIO/Utility.h>

#define LOG_TAG "SGBM"


//----------------------------------
// SGM-based stereo matching
//
// - HIGH_LEVEL_API: setero is evaluated by a single
//   nvxSemiGlobalMatchingNode
//
//---------------------------------

namespace hlsgm
{
    // This implementation uses an nvxSemiGlobalMatchingNode to evaluate
    // stereo.  This is the most simple and straight-forward way to do this
    class SGBM : public StereoMatching
    {
    public:
	SGBM(vx_context context, const StereoMatchingParams& params,
	     vx_image left, vx_image right, vx_image disparity);
	~SGBM();

	virtual void run();

	void printPerfs() const;

    private:
	vx_graph main_graph_;
	vx_node left_cvt_color_node_;
	vx_node right_cvt_color_node_;
	vx_node semi_global_matching_node_;
	vx_node convert_depth_node_;

    };

    void SGBM::run()
    {
	NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );
    }

    SGBM::~SGBM()
    {
	vxReleaseGraph( &main_graph_ );
    }

    SGBM::SGBM(vx_context context, const StereoMatchingParams& params,
		vx_image left, vx_image right, vx_image disparity)
	: main_graph_(nullptr)
    {
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_uint32 width = 0;
	vx_uint32 height = 0;

	NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
	NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
	NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

	main_graph = vxCreateGraph(context);
	NVXIO_CHECK_REFERENCE(main_graph_);

	// Convert images to grayscale
	vx_image left_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
	NVXIO_CHECK_REFERENCE(left_gray);

	vx_image right_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
	NVXIO_CHECK_REFERENCE(right_gray);

	left_cvt_color_node_ = vxColorConvertNode(main_graph_, left, left_gray);
	NVXIO_CHECK_REFERENCE(left_cvt_color_node_);

	right_cvt_color_node_ = vxColorConvertNode(main_graph_, right, right_gray);
	NVXIO_CHECK_REFERENCE(right_cvt_color_node_);

	// evaluate stereo
	vx_image disparity_short = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_S16);
	NVXIO_CHECK_REFERENCE(disparity_short);

	//
        // The SGM algorithm is now added as a node to the graph via the
        // nvxSemiGlobalMatchingNode().The input to the SGM node is previosuly
        // constructed left_gray and right_gray vx_images and the configuration
        // parameters. The output of the SGM node is the disparity_short image
        // that holds S16 fixed-point disparity values. The fixed-point values
        // have Q11.4 format (one sign bit, eleven integer bits and four
        // fractional bits). For the ease of further processing, we convert the
        // disparity map from fixed-point representation to U8 disparity
        // image. To do this, we drop the 4 fractional bits by right-shifting
        // the S16 values and then simply scale down the bit-width precision via
        // the vxConvertDepthNode().
        //
	semi_global_matching_node_ = nvxSemiGlobalMatchingNode(
	     main_graph_,
	     left_gray,
	     right_gray,
	     disparity_short,
	     params.min_disparity,
	     params.max_disparity,
	     params.P1,
	     params.P2,
	     params.sad,
	     params.ct_win_size,
	     params.hc_win_size,
	     params.bt_clip_value,
	     params.max_diff,
	     params.uniqueness_ratio,
	     params.scanlines_mask,
	     params.flags);
	NVXIO_CHECK_REFERENCE(semi_global_matching_node_);

	// Convert disparity from fixed point to grayscale
	vx_int32 shift = 4;
	vx_scalar s_shift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
	NVXIO_CHECK_REFERENCE(s_shift);
	convert_depth_node_ = vxConvertDepthNode(main_graph_, disparity_short, disparity, VX_CONVERT_POLICY_SATURATE, s_shift);
	vxReleaseScalar(&s_shift);
	NVXIO_CHECK_REFERENCE(convert_depth_node_);

	// verify the graph
	NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_));

	// Clean up
	vxReleaseImage(&left_gray);
	vxReleaseImage(&right_gray);

	vxReleaseImage(&disparity_short);

    }

    void SGBM::printPerfs() const
    {
	nvxio::printPerf(main_graph_, "Stereo");
	nvxio::printPerf(left_cvt_color_node_, "Left Color Convert");
	nvxio::printPerf(right_cvt_color_node_, "Right Color Convert");
	nvxio::printPerf(semi_global_matching_node_, "SGBM");
	nvxio::printPerf(convert_depth_node_, "Convert Depth");
    }
}	
