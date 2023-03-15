'''
[Descriptions]

아래의 Torchy 클래스는 img 폴더 내에서 사진 10장에 대한 object detection(객체 감지)를 수행합니다.
설계 및 로직 실행의 순서는 다음과 같습니다.

[TODO - In Progress - Done]

1. FasterrCnn 모델을 구축하고 훈련시키기 or 훈련된 FasterrCnn 모델을 불러와서 모델 형성하기
2. Pytorch로 작성한 DNN 모델을 onnx format으로 export 해서 파일로 저장
3. 저장한 onnx 파일을 ONNX Runtime으로 수행 
4. 이미지 10장으로 inference 수행해서 object detection 결과 box들을 이미지에 함께 출력 + 각각의 inference 수행시간 출력

[Limitations]

1. 현재 기존 모델의 학습이 끝난 후에 onnx 파일 포맷으로 export까지 성공하였으나, onnx runtime에서 수행하는 방법에서 Blocked 된 상황
2. Tagging 및 모델을 빌드 할 때 조금 더 전문적이고 세밀한 architecture 설계가 필요
3. 속도와 정확성 측면에서 upgrade 해야 할 부분에 대한 research

'''

# 필요한 라이브러리 import하기 (alphabet 순으로 정렬)
import cv2 
import datetime
import json
import numpy as np
import onnx
import onnxruntime
import os
import re
import torch
import torchvision

from torchvision import transforms as T
from PIL import Image 
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import *

# 사물을 인식을 위해서 Fasterrcnn 모델을 구축하며 형성한다.
class FasterrCnnModel(QtWidgets.QWidget):
    
    command = QtCore.pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.current_path = os.path.realpath(__file__)
        self.img_file_dir = None
        self.model = None
        self.device = None
        self.img = None
        self.transformed_img = None
        self.pred = None
        self.bboxes = None
        self.labels = None
        self.scores = None
        self.valid_obj_num = None
        self.class_name = None
        self.dummy_input = None
        self.start_time = None 
        self.end_time = None
        self.total_execute_time = None
    
    # 이미지 추론 작업을 이제부터 순차적으로 진행하며, 그 시작 시간을 측정합니다.
    def start_inference_job(self):
        self.start_time = datetime.datetime.now()
        
    # UI/User로 부터 입력받은 이미지의 이름을 파싱한 뒤에, 맨 앞의 index 넘버로, dataset 폴더 밑에 해당 이미지를 로드한다
    def get_target_img(self, img_file_name):
        index = re.findall(r'\d+', img_file_name)[0]
        self.img_file_dir = os.path.normpath(os.path.join(self.current_path, "../dataset/test_dataset_{}/img/{}.jpg".format(index,index)))

    # DNN 모델 中 대표적인 fasterrcnn 모델을 불러와서 셋업한다
    def set_model(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # 모델의 evaluation 과정에서 사용하지 않는 layer을 off 시키는 기능 (Dropout, Batchnorm) -> 추론 모드로 전환하기
        self.model.eval()
    
    # 기본적으로 CPU 엔진을 사용하지만, GPU을 사용할 수 있으면, GPU에서 모델을 사용합니다.
    def move_model_to_cpu_or_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        
    # pillow lib로 해당 target 이미지의 원본 파일에 대한 정보를 object 형식으로 불러오기
    def open_target_img(self):
        self.img = Image.open(self.img_file_dir)
    
    # 필요에 따라서 target img의 사이즈를 재조절 해주는 기능추가
    def resize_opened_target_img(self):
        resize = T.Resize([224, 224])
        self.img = resize(self.img)
    
    # 2D self.img을 3D 텐서 object로 transform 하고 CPU(GPU)로 옮기기
    def transform_img_to_tensor(self):
        transform = T.ToTensor()
        self.transformed_img = transform(self.img).to(self.device)
    
    # 변환된 이미지 객체를 model에 train 시켜서 예측값 얻어오기
    def predict_model_values(self):
        # PyTorch의 autograd engine(gradient을 계산해주는 context)을 비활성화 시켜서, 더이상 gradient을 트래킹 하지 않게 됨 --> 필요한 메모리 save + 연산속도 증가
        with torch.no_grad():
            self.pred = self.model([self.transformed_img])
        self.model.train()
    
    # 이미지 객체들의 테두리 박스 텐서, tag(label) 텐서, 그리고 (confidence) score텐서 element을 반환
    # descending order로서 score가 정렬되며, 각 labels list의 element는 labels.json의 index을 나타내며, 그와 매칭되는 score의 유사 confidence을 보여준다.
    def check_pred_info_status(self):
        print(self.pred)
        print(self.pred[0])
        print(self.pred[0].keys())
    
    # 예측되어서 나온 데이터를 각각의 element obj에 할당하기
    def assign_essential_components(self):
        self.bbox_vals, self.label_vals, self.score_vals = self.pred[0]['boxes'], self.pred[0]['labels'], self.pred[0]['scores']
    
    # .argwhere은 부등호 비교를 만족하는 torch의 텐서 index만 return 되며, .shape[0]을 통해서 해당 식에 만족하는 torch tensor 객체의 수를 가져온다.
    def get_valid_obj_num(self, threshold):
        self.valid_obj_num =  torch.argwhere(self.score_vals > threshold).shape[0]
        return self.valid_obj_num
    
    # 해당 테스트 데이터셋의 labels.json 데이터들을 읽어온다 (이유: 이미지에 labeling을 하기 위해서)
    def get_labels(self):
        labels_dir = os.path.normpath(os.path.join(self.img_file_dir, "../../label_info/labels.json"))
        with open(labels_dir) as f:
            data = json.load(f)
            f.close()
        self.labels = data
        
        return self.labels
    
    # 유효하게 detect된 object의 label을 중심으로 class_name과 bbox의 좌표값을 읽어오며, label_bounding_box와 label_name을 입력해준다.
    def detect_obj_bound_boxes(self):
        igg = cv2.imread(self.img_file_dir)
        font = cv2.FONT_HERSHEY_COMPLEX

        for i in range(self.valid_obj_num):
            x1, y1, x2, y2 = self.bbox_vals[i].numpy().astype("int")
            
            self.class_name = self.labels['label_names'][self.label_vals.numpy()[i]-1]
            
            # 사각형 bounding box의 좌표를 할당해서 두께가 1인 초록색 테두리 박스를 생성
            igg = cv2.rectangle(igg, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # 매칭되는 class name을 좌측 상단에 파랑색으로, FONT_HERSHEY_COMPLEX font을 사용해서 명시한다.
            igg = cv2.putText(igg, self.class_name , (x1, y1-10), font, 0.5, (255, 0, 0), 1 , cv2.LINE_AA)

        # 완성된 cv2 객체를 waitKey안의 miliseconds 만큼 보여준다
        cv2.imshow("object detected image",igg)
        cv2.waitKey(5000)
        # cv2.waitKey()

    # 생성된 모델을 onnx 포멧으로 export하기
    def export_model_to_onnx_format(self, exported_onnx_file_name):
        # dummpy_input 만들기 (1개의 element: dimension 0, 2개의 element: dimension 1 ...)
        self.dummy_input = torch.randn(1, 3, 224, 224)
        input_names = ["actual_input"]
        output_names = ["output"]

        # onnx 파일 export을 위해서 필요한 편수 넣어주기
        torch.onnx.export(self.model,
                        self.dummy_input,
                        exported_onnx_file_name,
                        verbose=False,
                        input_names=input_names,
                        output_names=output_names,
                        export_params=True,
                        )
    
    # model이 잘 convert 되었는지 확인하기 ([Fail]의 경우에는 에러 메세지 출력하기)
    def check_converted_model_validity(self, exported_onnx_file_name):
        try:
            onnx_model = onnx.load(exported_onnx_file_name)
            onnx.checker.check_model(onnx_model)
        except:
            print("[VALIDITY CHECK ERROR]: There is an invalidity in your exported onnx file!")

    # 이미지 추론 작업을 마치며 그 종료 시간을 측정합니다.
    def end_inference_job(self):
        self.end_time = datetime.datetime.now()
        self.total_execute_time = self.end_time - self.start_time
        self.sendCommand(self.total_execute_time)

    # 총 작업 소요시간을 (부모) 클래스인 GUI 클래스에 pyqtSlot signal을 통해서 emit(전송) 해줍니다.
    @QtCore.pyqtSlot()
    def sendCommand(self, time):
        self.command.emit(str(time))
        
        
    '''
    [TODO - In Progress - Blocked]
    : 아래부분의 코드는 onnxruntime 환경에서 onnx 파일 실행을 성공하고 나서 동작할 수 있는 코드입니다. 현재 Blocked.

    # 텐서 포맷을 다시 numpy로 바꿔주기
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # onnx 포멧으로 변환된 파일을 onnx 런타임에서 실행하기
    def execute_converted_model_in_onnx_format(self, exported_onnx_file_name):
        resize = T.Resize([224, 224])
        img = resize(self.img)

        img_ycbcr = img.convert('YCbCr')
        img_y, img_cb, img_cr = img_ycbcr.split()

        to_tensor = T.ToTensor()
        img_y = to_tensor(img_y)
        img_y.unsqueeze_(0)
        
        ort_session = onnxruntime.InferenceSession(exported_onnx_file_name)
        ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(img_y)}
        print(np.shape(ort_inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        print(np.shape(ort_inputs))
        img_out_y = ort_outs[0]

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교하기
    np.testing.assert_allclose(self.to_numpy(self.model), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    '''
