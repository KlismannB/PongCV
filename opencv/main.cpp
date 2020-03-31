#include "opencv4/opencv2/objdetect.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/videoio.hpp"
#include <iostream>
#include <cstdlib>

static int *pPlacar;
static int placarMaximo;
static int bateu = 0;

using namespace std;
using namespace cv;

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale);


int velocidadex();
int velocidadey();

void menu(Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale);

void menuFinal(Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale);

void writeoroni();
void readoroni();

string cascadeName;
string nestedCascadeName;
Mat fruta;

/**
 * @brief Draws a transparent image over a frame Mat.
 *
 * @param frame the frame where the transparent image will be drawn
 * @param transp the Mat image with transparency, read from a PNG image, with the IMREAD_UNCHANGED
   @param yPosi position of the frame image where the image will start. y position of the frame image where the image will start.
 */
void drawTransparency(Mat frame, Mat transp, int xPosi, int yPosi) {
    Mat mask;
    vector<Mat> layers;
    
    split(transp, layers); // seperate channels
    Mat rgb[3] = { layers[0],layers[1],layers[2] };
    mask = layers[3]; // png's alpha channel used as mask
    merge(rgb, 3, transp);  // put together the RGB channels, now transp insn't transparent
    transp.copyTo(frame.rowRange(yPosi, yPosi + transp.rows).colRange(xPosi, xPosi + transp.cols), mask);
}

void drawTransparency2(Mat frame, Mat transp, int xPosi, int yPosi) {
    Mat mask;
    vector<Mat> layers;
    
    split(transp, layers); // seperate channels
    Mat rgb[3] = { layers[0],layers[1],layers[2] };
    mask = layers[3]; // png's alpha channel used as mask
    merge(rgb, 3, transp);  // put together the RGB channels, now transp insn't transparent
    Mat roi1 = frame(Rect(xPosi, yPosi, transp.cols, transp.rows));
    Mat roi2 = roi1.clone();
    transp.copyTo(roi2.rowRange(0, transp.rows).colRange(0, transp.cols), mask);
    printf("%p, %p\n", roi1.data, roi2.data);
    double alpha = 0.9;
    addWeighted(roi2, alpha, roi1, 1.0 - alpha, 0.0, roi1);
}

int main( int argc, const char** argv )
{
    pPlacar = &placarMaximo;
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;
    
    fruta = cv::imread("laranja.png", IMREAD_UNCHANGED);
    if (fruta.empty())
        printf("Error opening file pong.pn\n");

    string folder = "/Users/edglaybarros/Documents/opencv-4.2.0/data/haarcascades/";
    cascadeName = folder + "haarcascade_frontalface_alt.xml";
    nestedCascadeName = folder + "haarcascade_eye_tree_eyeglasses.xml";
    inputName = "/dev/video0";

    if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!cascade.load(samples::findFile(cascadeName)))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if(!capture.open(0))
    {
        cout << "Capture from camera #" <<  inputName << " didn't work" << endl;
        return 1;
    }

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        readoroni();
       char c, ch;
                for(;;){
                capture >> frame;
                    flip(frame, frame,1);
                    if( frame.empty() )
                        break;
                
                menu(frame, cascade, nestedCascade, scale);
                c = (char)waitKey(10);
                if(c == 13 || c == 'q' || c == 'Q' || c == 27)
                break;
                }

                if(c == 13){
                for(;;)
                {
                    capture >> frame;
                    flip(frame, frame,1);
                    if( frame.empty() )
                        break;

                    //Mat frame1 = frame.clone();
                    detectAndDraw( frame, cascade, nestedCascade, scale);

                    ch = (char)waitKey(10);
                    if( ch == 13 || ch == 'q' || ch == 'Q' ){
                        
                        for(;;){
                            menuFinal(frame, cascade, nestedCascade, scale);
                            
                            char cha = (char)waitKey(10);
                            
                            if( cha == 13 || cha == 'q' || cha == 'Q' )
                            break;
                        }

                        writeoroni();
                        break;
                    }
                }
                }
    }

    return 0;
}


int velocidadex(){
    int spd, randito = rand() % 5;

    switch(randito){
        case 0:
            spd = 10;
            break;
        case 1:
            spd = 12;
            break;
        case 2:
            spd = 15;
            break;
        case 3:
            spd = 18;
            break;
        case 4:
            spd = 20;
            break;
    }

    if(rand() % 2 == 0)
    spd = - spd;

    return spd;
}

int velocidadey(){
    int spd, randito = rand() % 4;

    switch(randito){
        case 0:
            spd = 5;
            break;
        case 1:
            spd = 7;
            break;
        case 2:
            spd = 10;
            break;
        case 3:
            spd = 12;
            break;
    }

    if(rand() % 2 == 0)
    spd = - spd;

    return spd;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale)
{
    
    static int frames = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.2, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );

    frames++;

    //f (frames % 30 == 0)
        //system("mplayer /usr/lib/libreoffice/share/gallery/sounds/kling.wav &");

    t = (double)getTickCount() - t;
//    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Scalar color = colors[i%8];

        Rect r = faces[i];
        printf( "[%3d, %3d]\n", r.x, r.y);
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;

        rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                   Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                   color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
    }

    static int xPos = 310, yPos = 230;
    static int xSpd = velocidadex(), yspd = velocidadey();


    static Point pos;
    static int placar1 = 0, placar2 = 0;
    if(xPos > 660){
        xPos = 310;
        yPos = 230;
        placar1++;

        system("som_do_gol.mpeg &");

        xSpd = velocidadex();
        yspd = velocidadey();
    }

    
     if(xPos < -20){
        xPos = 360;
        yPos = 230;
        placar2++;

        system("som_do_gol.mpeg &");

        xSpd = velocidadex();
        yspd = velocidadey();
    }
    
    if(placar1 > placarMaximo){
        placarMaximo = placar1;
        bateu = 1;
    }
    else if(placar2 > placarMaximo){
        placarMaximo = placar2;
        bateu = 1;
    }
    if(yPos > 460 || yPos < 20){
           yspd = - yspd;
    }
    
    xPos+=xSpd;
    yPos+=yspd;
 
    pos.x = xPos;
    pos.y = yPos;
    
    //printf("%d, %d, %d, %d", xPos, yPos, pos.x, pos.y);

    circle( img, pos, 20, Scalar(255,255,0), -1, 8, 0 );

    char str[40];
    sprintf(str,"Player 1:%d    Player 2:%d  Record:%d",placar1, placar2, placarMaximo);
    cv::putText(img, //target image
        str, //text
        cv::Point(40, 50), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.0,
        CV_RGB(255, 255, 255), //font color
        3);

    imshow( "result", img );
    const int margem = 8;

    for (size_t i = 0; i < faces.size(); i++){
        Rect r = faces[i];
        if((xPos - 15 > cvRound(r.x) && xPos < cvRound(r.x) + cvRound(r.width))
        && ((yPos + 20 < cvRound(r.y) + margem) && (yPos + 20 > cvRound(r.y) - margem))){
            //a bolinha vem de cima para baixo
            if(yspd > 0){
            xSpd = -xSpd;
            yspd = -yspd;
            }

        }else if((xPos - 15 > cvRound(r.x) && xPos < cvRound(r.x) + cvRound(r.width))
        && ((yPos - 20 < cvRound(r.y) + cvRound(r.height) + margem) && (yPos - 20 > cvRound(r.y) + cvRound(r.height) - margem))){
            //a bolinha vem de baixo para cima
            if(yspd < 0){
            xSpd = -xSpd;
            yspd = -yspd;
            }
            
        }
          
          printf("%d  %d  %d  %d", r.x, r.y, r.width, r.height);

        if((yPos + 20 > cvRound(r.y) && yPos - 20 < cvRound(r.y) + cvRound(r.height))
         && ((xPos + 20 < cvRound(r.x) + margem) && (xPos + 20 > cvRound(r.x) - margem))){
            //entrara nesse if se a bolinha estiver vindo da esquerda para a direita
            if(xSpd > 0)
            xSpd = - xSpd;//garantia que a bolinha so seja invertida uma vez

        }else if(( yPos + 20 > cvRound(r.y) && yPos - 20 < cvRound(r.y) + cvRound(r.height))
        && ((xPos - 20 < cvRound(r.x) + cvRound(r.width) + margem) && (xPos - 20 > cvRound(r.x) + cvRound(r.width) - margem))){
            //entrara nesse if se a bolinha estiver vindo da direita para a esquerda
            if(xSpd < 0)
            xSpd = - xSpd;
          
        }
    }
}

void menu(Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale)
{
    //Menu inicial do jogo
    char str[40];
    sprintf(str,"Bem-vindo ao Pong CV!");
    cv::putText(img, //target image
        str, //text
        cv::Point(40, 50), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.4,
        CV_RGB(255, 255, 255), //font color
        3);

    sprintf(str,"Aperte Enter para jogar");
    cv::putText(img, //target image
        str, //text
        cv::Point(100, 150), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.0,
        CV_RGB(255, 255, 0), //font color
        3);

    sprintf(str,"Recorde de mais pontos: %d", placarMaximo);
    cv::putText(img, //target image
        str, //text
        cv::Point(100, 230), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.0,
        CV_RGB(255, 100, 0), //font color
        3);

    cv::putText(img, //target image
        "Aperte q ou Enter para fechar durante o jogo", //text
        cv::Point(50, 400), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        0.7,
        CV_RGB(50, 255, 0), //font color
        3);

    imshow( "result", img );
    
}

void menuFinal(Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale)
{   char str[40];
    sprintf(str,"Obrigado por jogar Pong CV!");
    cv::putText(img, //target image
        str, //text
        cv::Point(40, 200), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.2,
        CV_RGB(255, 255, 255), //font color
        3);
    if(bateu){
    sprintf(str,"Vc bateu o recorde!! Novo: %d", placarMaximo);
    cv::putText(img, //target image
        str, //text
        cv::Point(40, 300), //top-left position
        cv::FONT_HERSHEY_DUPLEX,
        1.0,
        CV_RGB(255, 255, 255), //font color
        3);
    }

    imshow( "result", img );
}
void readoroni(){
    // Leitura de arquivo para a variável recorde
    FILE *fp;
    
    fp = fopen("file.txt", "r");
    if(fp == NULL){
        puts("ERRO AO ABRIR fp.");
    }else{
        fscanf(fp, "%d%*c", pPlacar);
    }
    
    fclose(fp);
}

void writeoroni(){
    //Escrita de arquivo para a variável recorde
        FILE *fp;
    
        fp = fopen("file.txt", "w+r");
    
        if (fp == NULL) puts("Erro ao abrir FP.");
        else fprintf(fp, "%d", placarMaximo);
}
