#include "/home/codeleaded/System/Static/Library/WindowEngine1.0.h"
#include "/home/codeleaded/System/Static/Library/GSprite.h"
#include "/home/codeleaded/System/Static/Library/NeuralNetwork.h"

#define SPRITE_PATH         "/home/codeleaded/Data/NN/Digits/"
#define DATA_PATH           "/home/codeleaded/Data/NN/DigitsGray/"
#define SPRITE_TEST         "testing"
#define SPRITE_TRAINING     "training"
#define SPRITE_COUNT        4
#define SPRITE_MAX          300

#define NN_PATH             "./data/Model.nnalx"
#define NN_COUNT            10
#define NN_LEARNRATE        0.1f

int epoch = 0;
int reality = 0;
int prediction = 0;
NeuralType loss = 0.0f;
GSprite sp;
AlxFont font;
NeuralNetwork nnet;

void NeuralNetwork_Render(NeuralNetwork* nn){
    for(int i = 0;i<nn->layers.size;i++){
        NeuralLayer* nl = (NeuralLayer*)Vector_Get(&nn->layers,i);
        
        for(int j = 0;j<nl->count;j++){
            const int dx = 400.0f;
            const int x = i * dx;
            const int y = j * font.CharSizeY * 3;

            RenderRect(x,y,100.0f,font.CharSizeY * 2,GREEN);
            
            String str = String_Format("%f",nl->values[j]);
            RenderCStrSizeAlxFont(&font,str.Memory,str.size,x,y,GRAY);
            String_Free(&str);
        
            if(nl->precount > 0){
                str = String_Format("%f",nl->biases[j]);
                RenderCStrSizeAlxFont(&font,str.Memory,str.size,x,y + font.CharSizeY,GRAY);
                String_Free(&str);
            }
        
            const int max = 3;
            const int count = nl->precount < max ? nl->precount : max;
            for(int k = 0;k<count;k++){
                if(nl->weights && nl->weights[j]){
                    str = String_Format("%f",nl->weights[j][k]);
                    RenderCStrSizeAlxFont(&font,str.Memory,str.size,x - dx * 0.5f,y + k * font.CharSizeY,GRAY);
                    String_Free(&str);
                }
            }
        }
    }
}

NeuralDataPair NeuralDataPair_Make_GSprite(char* path,int number,int item){
    CStr ntraining_s = CStr_Format("%s/%d/%d.sprg",path,number,item);
    GSprite sp = GSprite_Load(ntraining_s);
    CStr_Free(&ntraining_s);
    
    NeuralType outs[NN_COUNT];
    memset(outs,0,sizeof(outs));
    outs[number] = 1.0f;

    NeuralDataPair ndp = NeuralDataPair_New(sp.img,outs,sp.w * sp.h,NN_COUNT);
    GSprite_Free(&sp);

    return ndp;
}
NeuralDataMap NeuralDataMap_Make_GSprite(char* path){
    NeuralDataMap ndm = NeuralDataMap_New();
    for(int i = 0;i<10;i++){
        for(int j = 0;j<SPRITE_COUNT;j++){
            NeuralDataPair ndp = NeuralDataPair_Make_GSprite(path,i,epoch + j);
            Vector_Push(&ndm,&ndp);
        }
    }
    epoch += SPRITE_COUNT;
    if(epoch + SPRITE_COUNT > SPRITE_MAX){
        epoch = 0;
    }
    printf("Epoch: %d\n",epoch);
    return ndm;
}

NeuralDataMap NeuralDataMap_Make_GSprite_R(char* path){
    NeuralDataMap ndm = NeuralDataMap_New();
    for(int i = 0;i<10;i++){
        for(int j = 0;j<SPRITE_COUNT;j++){
            NeuralDataPair ndp = NeuralDataPair_Make_GSprite(path,i,Random_u32_MinMax(0,SPRITE_MAX));
            Vector_Push(&ndm,&ndp);
        }
    }
    return ndm;
}

void Setup(AlxWindow* w){
    RGA_Set(Time_Nano());

    sp = GSprite_None();
    font = AlxFont_MAKE_HIGH(12,24);
    
    if(Files_isFile(NN_PATH))
        nnet = NeuralNetwork_Load(NN_PATH);
    else
        nnet = NeuralNetwork_Make((unsigned int[]){ 784,16,10,0 });
}
void Update(AlxWindow* w){
    if(Stroke(ALX_KEY_W).PRESSED){
        NeuralDataMap ndm = NeuralDataMap_Make_GSprite(DATA_PATH SPRITE_TRAINING);
        NeuralNetwork_Learn(&nnet,&ndm,NN_LEARNRATE);
        NeuralDataMap_Free(&ndm);

        ndm = NeuralDataMap_Make_GSprite(DATA_PATH SPRITE_TEST);
        loss = NeuralNetwork_Test_C(&nnet,&ndm);
        NeuralDataMap_Free(&ndm);
    }
    if(Stroke(ALX_KEY_S).PRESSED){
        unsigned int ndir = Random_u32_MinMax(0,10);
        unsigned int item = Random_u32_MinMax(0,SPRITE_MAX);

        NeuralDataPair ndp = NeuralDataPair_Make_GSprite(DATA_PATH SPRITE_TEST,ndir,item);
        loss = NeuralNetwork_Test(&nnet,&ndp);
        NeuralDataPair_Free(&ndp);

        prediction = NeuralNetwork_Decision(&nnet);
        reality = ndir;

        CStr ntraining_s = CStr_Format("%s/%d/%d.sprg",DATA_PATH SPRITE_TEST,ndir,item);
        GSprite_Free(&sp);
        sp = GSprite_Load(ntraining_s);
        CStr_Free(&ntraining_s);
    }
    if(Stroke(ALX_KEY_Q).PRESSED){
        NeuralNetwork_Save(&nnet,NN_PATH);
        printf("[NeuralNetwork]: Save -> Success!\n");
    }
    if(Stroke(ALX_KEY_E).PRESSED){
        NeuralNetwork_Free(&nnet);
        if(Files_isFile(NN_PATH)){
            nnet = NeuralNetwork_Load(NN_PATH);
            printf("[NeuralNetwork]: Load -> Success!\n");
        }else{
            nnet = NeuralNetwork_Make((unsigned int[]){ 784,16,10,0 });
            printf("[NeuralNetwork]: Load -> Failed!\n");
        }
    }

    Clear(DARK_BLUE);

    GSprite_Render(WINDOW_STD_ARGS,&sp,GetWidth() - sp.w - 50.0f,0.0f);

    NeuralNetwork_Render(&nnet);

    //String str = String_Format("T:%d, ND:%d, I:%d",test,ndir,item);
    //RenderCStrSize(str.Memory,str.size,0.0f,0.0f,WHITE);
    //String_Free(&str);

    String str = String_Format("Loss: %f, Is: %d, Pre: %d, -> %s",loss,reality,prediction,(reality == prediction ? "correct" : "wrong"));
    RenderCStrSize(str.Memory,str.size,0.0f,GetHeight() - GetAlxFont()->CharSizeY,WHITE);
    String_Free(&str);
}
void Delete(AlxWindow* w){
    NeuralNetwork_Free(&nnet);
    GSprite_Free(&sp);
    AlxFont_Free(&font);
}

int main(){
    if(Create("RGB to G",1920,1080,1,1,Setup,Update,Delete))
        Start();
    return 0;
}