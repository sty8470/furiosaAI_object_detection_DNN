'''
[Prerequisites]
우선, reference 폴더 안에 있는 pptx 파일을 읽고, 전체 project에 대한 overview을 할 수 있기를 권장드립니다.
다음으로는, requirements.txt에 있는 해당 라이브러리들의 버젼에 맞는 설치 및 환경셋업을 해야 호환성의 에러 없이 프로그램이 잘 동작할 수 있습니다.

[Descriptions]

아래의 MainGUI 클래스는 detect_objects.py 파일에 있는 Object Detection 모델인 Fasterrcnn의 클래스의 동작에 대한 
입력 input과 결과 output을 사용자에게 보다 더 편리하고 쉽게 보여주는 GUI 인터페이스 입니다. 

[TODO - In Progress - Done]

현재 총 10가지 사진 中 1개를 선택 후 `사물인식 시작하기` 버튼을 눌러서 순차적으로 할 수 있으며, 
`프로그램 재시작`을 누르면, 새로운 GUI 인스턴스가 생성되어서, 다른 이미지에 대한 `사물인식`을 수행할 수 있습니다.
(단, 한 thread가 돌아가는 中에는 사용불가 합니다)

[Limitations]

1. 사물인식이 진행되는 time span에서도 dynamic한 방식으로 stopwatch의 소요시간이 1초씩 increment 하는 방식을 처음에 구현하였지만, 
multi-threading issue로 현재 Blocked된 상황
2. 사물 인식이 진행되는 상황을 조금 더 세부적으로 잘 보여주거나, 도중에 stop 및 restart 할 수 있는 고급기능에 대한 향상은 추가 research 필요

'''

# 관련되 라이브러리와 모듈들을 가져오기
import sys
import os

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from detect_objects import FasterrCnnModel

# 시스템 환경변수에 현재 파일 경로 넣어주기 (빌드시에 모듈 연결성과 안전성을 확보할 수 있다)
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
sys.path.append(os.path.normpath(os.path.join(current_path, '../../')))

# Object Detection을 유저가 쉽고 선택하기 편리하게 실행할 수 있게 GUI 클래스 생성하기
class MainGUI(QDialog):
    
    # 생성자에서 백엔드 로직 클래스인 FasterCnnModel을 instantiate하고 signal 객체를 서로 연결시키기
    def __init__(self):
        super().__init__()
        self.f = FasterrCnnModel()
        self.f.command.connect(self.set_final_timer_val)
        self.init_gui()
    
    # GUI의 전체적인 디자인과 골조를 형성하는 부분
    def init_gui(self):
        
        # 다이얼로그 UI의 아이콘 설정 및 제목 설정하기
        self.setWindowIcon(QIcon(os.path.join(current_path, './icon_img/google.jpg')))
        self.setWindowTitle('Object Detection Model (사물 인식하기)')
        
        # 이미지 종류 및 최종소요시간과 관련된 위젯의 layout 정렬하기
        self.main_v_layout = QVBoxLayout()
        self.search_h_layout = QHBoxLayout()
        self.final_time_taken_h_layout = QHBoxLayout()
        
        # 각 element의 라벨과 콤보박스에 들어갈 내용을 기술하고 객체를 생성하기
        self.search_label = QLabel("이미지 종류: ")
        self.final_time_taken_label = QLabel("최종 소요시간은:")
        self.final_time_taken_value = QLabel()
        self.search_label.setFixedWidth(100)
        self.search_combo_box = QComboBox()
        self.search_combo_box.setFixedWidth(250)
        self.search_combo_box.addItem("1_대학교_동아리_사진.jpg")
        self.search_combo_box.addItem("2_우정_친구_사진.jpg")
        self.search_combo_box.addItem("3_개와_고양이_사진.jpg")
        self.search_combo_box.addItem("4_여성_개_고양이_사진.jpg")
        self.search_combo_box.addItem("5_아이와_오리_사진.jpg")
        self.search_combo_box.addItem("6_리빙_가구_사진.jpg")
        self.search_combo_box.addItem("7_고속도로_차_사진.jpg")
        self.search_combo_box.addItem("8_하늘_철새_사진.jpg")
        self.search_combo_box.addItem("9_탁구채_탁구공_사진.jpg")
        self.search_combo_box.addItem("10_하늘_에어벌룬_사진.jpg")
        
        # 각 element들을 layout에 넣어주고, alignment을 맞춰주기
        self.search_h_layout.addWidget(self.search_label)
        self.search_h_layout.addWidget(self.search_combo_box)
        self.search_h_layout.setAlignment(Qt.AlignLeft)
        self.final_time_taken_h_layout.addWidget(self.final_time_taken_label)
        self.final_time_taken_h_layout.addWidget(self.final_time_taken_value)
        self.final_time_taken_h_layout.setAlignment(Qt.AlignLeft)
        
        # 하단의 제출라인에서 "사물인식 실행", "프로그램 재시작", 그리고 "취소" 버튼을 생성 및 기능연결을 위한 셋업하기
        self.submission_h_layout = QHBoxLayout()
        self.execute_button = QPushButton("사물인식 시작하기")
        self.restart_button = QPushButton("프로그램 재시작")
        self.cancel_button = QPushButton("취소")
        self.execute_button.clicked.connect(self.submit)
        self.restart_button.clicked.connect(self.restart_program)
        self.cancel_button.clicked.connect(self.close)
        
        # 하단의 제출라인에서 위젯들을 layout에 넣어서 저장해주기
        self.submission_h_layout.addWidget(self.execute_button)
        self.submission_h_layout.addWidget(self.restart_button)
        self.submission_h_layout.addWidget(self.cancel_button)
        
        # GUI의 다이얼로그의 전체 layout을 최종적으로 완결해 주는 부분
        self.main_v_layout.addLayout(self.search_h_layout)
        self.main_v_layout.addLayout(self.final_time_taken_h_layout)
        self.main_v_layout.addLayout(self.submission_h_layout)
        self.main_v_layout.setContentsMargins(20, 10, 20, 10)
        self.setLayout(self.main_v_layout)
        self.setGeometry(300, 300, 400, 150)
        self.showDialog()

    # "사물인식 시작하기" 버튼이 눌렀을 때 구동되는 함수들 -> FasterrCnn class에 정의된 Deep Learning method들이 순차적으로 실행됨 (90% confidence로 metric 측정)
    def submit(self):
        print('해당 이미지 사물 객체 인식을 시작합니다!!')
        print('=====================================================================================================')
        self.f.start_inference_job()
        self.f.get_target_img(self.search_combo_box.currentText())
        self.f.set_model()
        self.f.move_model_to_cpu_or_gpu()
        self.f.open_target_img()
        # self.f.resize_opened_target_img()      ### 필요에 따라서 이미지 사이즈 크기 재조절 가능 ###
        self.f.transform_img_to_tensor()
        self.f.predict_model_values()
        self.f.check_pred_info_status()
        self.f.assign_essential_components()
        self.f.get_valid_obj_num(0.9)
        self.f.get_labels()
        self.f.detect_obj_bound_boxes()
        exported_onnx_file_name = "test_"+self.search_combo_box.currentText().split('.')[0]+".onnx"
        self.f.export_model_to_onnx_format(exported_onnx_file_name)
        self.f.check_converted_model_validity(exported_onnx_file_name)
        # f.execute_converted_model_in_onnx_format()   ### TODO 추후에 구현 예정 및 논의 필요: 현재는 X ###
        self.f.end_inference_job()
        print('=====================================================================================================')
        print('모델을 생성하고 이미지 객체를 추론하는데 걸린 시간은 총 {}입니다'.format(self.f.total_execute_time))
    
    # "프로그램 재시작" 버튼이 눌러지면, 기존의 QApplication은 종료되고 새로운, GUI object가 생성됨
    def restart_program(self):
        self.done(0)
        win = MainGUI()
    
    # FasterrCnn 모델에서 모든 연산(사물 인식)이 끝나고 걸린 최종 수행시간을 get하는 pyqt 시그널의 slot 부분
    # 이후 string parsing을 통한 초 단위 (걸린 시간) 계산하기
    @QtCore.pyqtSlot(str)
    def set_final_timer_val(self, time):
        self.minutes = time.split(':')[1]
        self.seconds = time.split(':')[2].split('.')[0]
        self.mili_seconds = time.split(':')[2].split('.')[1]
        
        # 현재는 60초 미만이 실행되어서 아래와 같이 string parsing 이후에 최종 수행 시간을 변환해주고, 
        if int(self.seconds) < 60:
            self.final_time_taken_value.setText("약 "+ str(round(float(time.split(':')[2]),2)) + "초 입니다.")
        
        # 만약에 60초가 초과한 경우에는, 아래와 같이 분과 초을 따로 parsing 하여서 출력해줌
        else:
            self.final_time_taken_value.setText("약 "+ self.minutes + "분 " + self.seconds + "초 입니다.")
    
    # 부모 클래스인 QDialog의 상속이후에 Dialog을 보여주는 부분        
    def showDialog(self):
        return super().exec_()

# 현재 파일에서만, QApplication을 실행해서 해당 GUI을 보여주기    
if __name__ == '__main__':
    app = QApplication([])
    win = MainGUI()