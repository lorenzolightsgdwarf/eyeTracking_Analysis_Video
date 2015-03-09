/* 
 * File:   main.cpp
 * Author: chili
 *
 * Created on March 2, 2015, 1:49 PM
 */

#include <cstdlib>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <chilitags/chilitags.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <set>
#define FIX_FILE_STARTCOL 1
#define FIX_FILE_STOPCOL 3
#define FIX_X 4
#define FIX_Y 5

bool gui = true;
using namespace std;

/*
 * argv[1] video
 * argv[2] event file
 * argv[3] reference image
 * argv[4] aoi image path
 * argv[5] aoi code path
 * argv[>=6] tags to not detect
 */

chilitags::Chilitags detector;
std::map<int, chilitags::Quad> refTags;
cv::Mat aoiImage;
std::map<int, string> beams, joints;
cv::Mat refImage;
int initReference(char*, char*, char*);
int totframe = 0, nullframe = 0;
set<int> invalidTag;
string trialname;

int main(int argc, char** argv) {

    for (int i = 0; i < argc; i++)
        cout << argv[i] << "\n";

    trialname = argv[1];
    trialname = trialname.substr(0, 8);


    detector.setFilter(0, 0.);
    detector.setPerformance(chilitags::Chilitags::ROBUST);

    if (argc > 6) {
        for (int i = 6; i < argc; i++) {
            invalidTag.insert(boost::lexical_cast<int>(argv[i]));
        }
    }

    if (initReference(argv[3], argv[4], argv[5]) < 0)
        return -1;


    cv::VideoCapture inputVideo(argv[1]);
    cv::Mat frame;
    ifstream inputEvent(argv[2]);
    if (!inputEvent.is_open()) {
        cerr << "open event file failed:" << argv[2];
        return -1;
    }
    ofstream outputEvent("AOI_" + boost::lexical_cast<string>(argv[2]));
    if (!outputEvent.is_open()) {
        cerr << "open output event file failed";
        return -1;
    }
    string line, header, primaryAOI = "", secondaryAOI = "";
    int line_usage = 0;
    getline(inputEvent, header);
    getline(inputEvent, line);
    outputEvent << header << ",primary AOI,secondary AOI\n";

    if (gui) {
        cv::namedWindow("Original Frame", CV_WINDOW_NORMAL);
        cv::namedWindow("Reference image", CV_WINDOW_NORMAL);
    }

    double currentTime = inputVideo.get(CV_CAP_PROP_POS_MSEC);
    while (inputVideo.read(frame)) {
        totframe++;
        bool is_processable = false;
        cv::Mat demoFrame = frame.clone();
        while (1) {
            vector <std::string> fields;
            boost::split(fields, line, boost::is_any_of(","));
            double start = boost::lexical_cast<double>(fields[FIX_FILE_STARTCOL]);
            double stop = boost::lexical_cast<double>(fields[FIX_FILE_STOPCOL]);
            if (currentTime > stop) {
                if (line_usage == 0)
                    nullframe++;
                outputEvent << line << "," << primaryAOI << "," << secondaryAOI << "\n";
                if (!getline(inputEvent, line))
                    goto end;
                primaryAOI = "";
                secondaryAOI = "";
                line_usage = 0;
                continue;

            } else if (currentTime >= start && currentTime <= stop)
                is_processable = true;

            break;
        }

        if (is_processable) {
            std::map<int, chilitags::Quad> tags = detector.find(frame, chilitags::Chilitags::DETECT_ONLY);
            if (tags.size() > 0) {
                std::vector<cv::Point2f> src, dst;
                for (const std::pair<int, chilitags::Quad> &tag : tags) {
                    int id = tag.first;
                    if (refTags.count(id) == 0) continue;
                    // We wrap the corner matrix into a datastructure that allows an
                    // easy access to the coordinates
                    src.push_back(cv::Point2f(tag.second(0, 0), tag.second(0, 1)));
                    src.push_back(cv::Point2f(tag.second(1, 0), tag.second(1, 1)));
                    src.push_back(cv::Point2f(tag.second(2, 0), tag.second(2, 1)));
                    src.push_back(cv::Point2f(tag.second(3, 0), tag.second(3, 1)));
                    dst.push_back(cv::Point2f(refTags[id](0, 0), refTags[id](0, 1)));
                    dst.push_back(cv::Point2f(refTags[id](1, 0), refTags[id](1, 1)));
                    dst.push_back(cv::Point2f(refTags[id](2, 0), refTags[id](2, 1)));
                    dst.push_back(cv::Point2f(refTags[id](3, 0), refTags[id](3, 1)));
                }
                if (src.size() == 0) continue;
                cv::Mat H = cv::findHomography(src, dst, CV_LMEDS);
                vector <std::string> fields;
                boost::split(fields, line, boost::is_any_of(","));
                cv::Mat frameGaze = (cv::Mat_<double>(3, 1) << boost::lexical_cast<double>(fields[FIX_X]),
                        boost::lexical_cast<double>(fields[FIX_Y]), 1);
                cv::Mat projGaze = H*frameGaze;

                projGaze.at<double>(0, 0) /= projGaze.at<double>(2, 0);
                projGaze.at<double>(1, 0) /= projGaze.at<double>(2, 0);
                projGaze.at<double>(2, 0) /= projGaze.at<double>(2, 0);

                if (projGaze.at<double>(0, 0) > 0 && projGaze.at<double>(0, 0) < aoiImage.cols
                        && projGaze.at<double>(1, 0) > 0 && projGaze.at<double>(1, 0) < aoiImage.rows) {

                    cv::Vec3b intensity = aoiImage.at<cv::Vec3b>(projGaze.at<double>(1, 0), projGaze.at<double>(0, 0));
                    int blue = (int) intensity.val[0];
                    int red = (int) intensity.val[2];
                    string currentAOI = "";
                    bool is_joint = false;
                    if (red > 0 && red < 255) {
                        line_usage++;
                        currentAOI = beams[red];
                    } else if (blue > 0 && blue < 255) {
                        is_joint = true;
                        line_usage++;
                        currentAOI = joints[red];
                    }
                    if (currentAOI.length() != 0) {
                        if (primaryAOI.length() == 0)
                            primaryAOI = currentAOI;
                        else {
                            if (is_joint) {
                                if (secondaryAOI.find(primaryAOI) == std::string::npos) {
                                    if (secondaryAOI.length() != 0)
                                        secondaryAOI = secondaryAOI + ":";
                                    secondaryAOI = secondaryAOI + primaryAOI;
                                }
                                primaryAOI = currentAOI;
                            } else {
                                if (secondaryAOI.find(currentAOI) == std::string::npos) {
                                    if (secondaryAOI.length() != 0)
                                        secondaryAOI = secondaryAOI + ":";
                                    secondaryAOI = secondaryAOI + currentAOI;
                                }
                            }

                        }
                    }
                }
                if (gui) {
                    cv::circle(demoFrame, cv::Point(frameGaze.at<double>(0, 0), frameGaze.at<double>(1, 0)), 20, cv::Scalar(255, 0, 0), 5);
                    cv::Mat demoReference = refImage.clone();
                    cv::circle(demoReference, cv::Point(projGaze.at<double>(0, 0), projGaze.at<double>(1, 0)), 20, cv::Scalar(255, 0, 0), -1);
                    cv::putText(demoReference, primaryAOI, cv::Point2d(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(255, 0, 0));
                    cv::imshow("Reference image", demoReference);

                }
            }
        }
        if (gui) {
            cv::putText(demoFrame, boost::lexical_cast<string>(currentTime), cv::Point2d(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1.f, cv::Scalar(255, 0, 0));
            cv::imshow("Original Frame", demoFrame);
            cv::waitKey(0);

        }
        currentTime = inputVideo.get(CV_CAP_PROP_POS_MSEC);
    }
end:
    cout << "Done!\n";
    ofstream statistics("Statistics_" + boost::lexical_cast<string>(argv[2]));
    statistics << "Total Frames: " << totframe << "\n";
    statistics << "Null Frames: " << nullframe << "\n";
    statistics << "Null Frames perc: " << 100 * (double) nullframe / (double) totframe << "\n";
    statistics.close();
    inputVideo.release();
    outputEvent.close();
    inputEvent.close();

    return 0;
}

int initReference(char* imgpath, char* aoiimgpath, char* aoicodepath) {
    refImage = cv::imread(imgpath, 1);
    refTags = detector.find(refImage, chilitags::Chilitags::DETECT_ONLY);
    for (int i : invalidTag) {
        if (refTags.count(i) > 0) {
            refTags.erase(i);
        }
    }
    aoiImage = cv::imread(aoiimgpath, 1);

    if (refImage.size != aoiImage.size) {
        cerr << "ref and aoi image are not of the  same size";
        return -1;
    }

    ifstream inputAOICode(aoicodepath);
    if (!inputAOICode.is_open()) {
        cerr << "aoicode file open failed:" << aoicodepath;
        return -1;
    }

    string line;
    while (getline(inputAOICode, line)) {
        vector <std::string> fields;
        boost::split(fields, line, boost::is_any_of(","));
        string aoiName = fields[0];
        //cout << fields[1] << " " << fields[2] << "\n";
        int r = boost::lexical_cast<int>(fields[1]);
        int b = boost::lexical_cast<int>(fields[2]);
        //Beam
        if (r > 0)
            beams[r] = aoiName;
        else
            joints[b] = aoiName;
    }

    inputAOICode.close();

    if (gui) {
        cv::Mat demoImg = refImage.clone();
        // First, we set up some constants related to the information overlaid
        // on the captured image
        const static cv::Scalar COLOR(255, 0, 255);
        // OpenCv can draw with sub-pixel precision with fixed point coordinates
        static const int SHIFT = 16;
        static const float PRECISION = 1 << SHIFT;

        for (const std::pair<int, chilitags::Quad> & tag : refTags) {

            int id = tag.first;
            // We wrap the corner matrix into a datastructure that allows an
            // easy access to the coordinates
            const cv::Mat_<cv::Point2f> corners(tag.second);

            // We start by drawing the borders of the tag
            for (size_t i = 0; i < 4; ++i) {
                cv::line(
                        demoImg,
                        PRECISION * corners(i),
                        PRECISION * corners((i + 1) % 4),
#ifdef OPENCV3
                        COLOR, 1, cv::LINE_AA, SHIFT);
#else
                        COLOR, 1, CV_AA, SHIFT);
#endif
            }

            // Other points can be computed from the four corners of the Quad.
            // Chilitags are oriented. It means that the points 0,1,2,3 of
            // the Quad coordinates are consistently the top-left, top-right,
            // bottom-right and bottom-left corners.
            // (i.e. clockwise, starting from top-left)
            // Using this, we can compute (an approximation of) the center of
            // tag.
            cv::Point2f center = 0.5f * (corners(0) + corners(2));
            cv::putText(demoImg, cv::format("%d", id), center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, COLOR);
        }
        cv::imshow("reference image", demoImg);
        cv::imshow("AOI image", aoiImage);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}