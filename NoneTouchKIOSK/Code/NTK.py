import sys
from PyQt5.QtWidgets import *
import threading
import time
import cv2 
import mediapipe as mp 
import numpy as np
import time
from scipy import stats
from pyzbar import pyzbar
from PyQt5.QtCore import Qt

"""https://github.com/KMS1105/AI-Computer-Study-Club/tree/main"""

barcode_data = ''
Detect_data = ''
run = True
QRrun = False

def decode_qr_code(frame):
    """QR 인식 함수"""
    global barcode_data
    barcodes = pyzbar.decode(frame)

    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        print(barcode_data, type(barcode_data))

def recog_gesture():
    """제스처 인식 함수"""
    global QRrun
    result_list = []
    max_num_hands = 1 # 손은 최대 1개만 인식
    kiosk_gesture = {
        0:'zero', 1:'one', 2:'two', 3:'five', 4:'ok', 5:'good'
    }

    mp_hands = mp.solutions.hands 

    hands = mp_hands.Hands(
        max_num_hands=max_num_hands, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7)

    file = np.genfromtxt('C:/Users/12612/OneDrive/바탕 화면/Python/AI/HPE/동아리/my_gesture_train.csv', delimiter=',') # 각 제스처들의 라벨과 각도가 저장되어 있음, 정확도를 높이고 싶으면 데이터를 추가해보자!**
    angle = file[:,:-1].astype(np.float32) # 각도
    label = file[:, -1].astype(np.float32) # 라벨
    knn = cv2.ml.KNearest_create() #K-최근접 알고리즘 생성
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        number = 0

        while True :
            if number > 3: 
                break

            ret, img = cap.read()
            if not ret:
                continue
            
            if QRrun == True:
                thread2 = threading.Thread(target=decode_qr_code(img)) #스레드 생성
                thread2.start()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 각도를 인식하고 제스처를 인식하는 부분
            if result.multi_hand_landmarks is not None: # 만약 손을 인식하면
                number += 1
                start = time.time()
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3)) # (joint수, (x,y,z))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z] # 각 joint마다 x,y,z 좌표 저장

                    # Compute angles between joints joint마다 각도 계산
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]관절벡터
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 벡터 정규화(크기 1 벡터) = v / 벡터의 크기

                    # Get angle using arcos of dot product **내적 후 arcos으로 각도를 구해줌**
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

                    angle = np.degrees(angle) # radian -> degree

                    # Inference gesture 학습시킨 제스처 모델에 참조를 한다.
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3) # k가 3일 때 값을 구한다!
                    idx = int(results[0][0]) # 인덱스를 저장!

                    # Draw gesture result
                    if idx in kiosk_gesture.keys():
                        cv2.putText(img, text=kiosk_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        rst = kiosk_gesture[idx].upper()
                        result_list.append(rst)

    mode = stats.mode(result_list)[0] #가장 보편적인 값 찾기

    if len(mode) == 0 : 
        return 'fail'
    
    else : 
        global Detect_data
        
        Detect_data = str(mode[0])
        print(Detect_data)

def last_fnc_detect():
    """0.1초마다 제스처 인식 함수 및 그에 따른 UI이벤트 실행"""
    while True:
        if MainWindow.exe_last_fnc_Detect == False:
            recog_gesture()
            MainWindow.fnc_Detect()
            time.sleep(0.1)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NTK')
        self.wid = 760
        self.hei = 950
        self.resize(self.wid, self.hei)
        self.center()
        self.nowpg = 0
        self.run = True
        self.topping = [['HOT', 0], ['ICE', 500]]
        self.basket = []
        self.exe_last_fnc_Detect = False
        self.select_menu = ""
        self.total_price = 0
        self.MenuText = ''
        self.OptionText = ''
        self.menu = [["아메리카노", 2500], ["카페라떼", 3000], ["에스프레소", 2000]]
        self.MainText = 'Shopping Basket: '
        self.new_text = ''

        self.stack = QStackedWidget(self) 

        # QStackedWidget 생성 
        self.stack.setGeometry(0,0,self.wid, self.hei) # 위치 및 크기 지정 
        self.stack.setFrameShape(QFrame.Box) # 테두리 설정(보기 쉽게) 

        # 입력할 page를 QWidget으로 생성 
        self.Mainpage = QWidget(self) 
        self.Optionpage = QWidget(self)
        self.Selectpage = QWidget(self) 
        self.Paypage = QWidget(self)
        self.Paypage2 = QWidget(self)
        self.Receiptpage = QWidget(self)

        # page에 위젯 추가하기 
        self.Prebtn1 = QPushButton(self.Optionpage)
        self.Prebtn1.setText('뒤로가기')
        self.Prebtn1.setGeometry(20,20,60,40)
        self.Prebtn2 = QPushButton(self.Receiptpage)
        self.Prebtn2.setText('뒤로가기')
        self.Prebtn2.setGeometry(20,20,60,40)
        self.Prebtn3 = QPushButton(self.Selectpage)
        self.Prebtn3.setText('뒤로가기')
        self.Prebtn3.setGeometry(20,20,60,40)

        self.Main_check = QLabel(self.Mainpage)
        self.Main_check.setFrameShape(QFrame.Box)  
        self.Main_check.setText('Shopping Basket: 0')
        self.Main_check.setGeometry(30, 720, 700, 200)
        self.btn1 = QPushButton(self.Mainpage) 
        self.btn1.setText(str("(1)\n"+self.menu[0][0])+'\n가격: '+str(self.menu[0][1]))
        self.btn1.setGeometry(20,160,200,200) 
        self.btn2 = QPushButton(self.Mainpage) 
        self.btn2.setText(str("(2)\n"+self.menu[1][0])+'\n가격: '+str(self.menu[1][1]))
        self.btn2.setGeometry(235,160,200,200) 
        self.btn3 = QPushButton(self.Mainpage) 
        self.btn3.setText(str("(3)\n"+self.menu[2][0])+'\n가격: '+str(self.menu[2][1]))
        self.btn3.setGeometry(450,160,200,200) 
        self.checkbtn1 = QPushButton(self.Mainpage)
        self.checkbtn1.setText('확인') 
        self.checkbtn1.setGeometry(660,870,60,40)
        self.Reset = QPushButton(self.Mainpage)
        self.Reset.setText('초기화') 
        self.Reset.setGeometry(15,15,60,40)

        self.option1 = QPushButton(self.Optionpage) 
        self.option1.setText("(1)\n"+'옵션1: \n'+self.topping[0][0]+'\n가격: '+str(self.topping[0][1])) 
        self.option1.setGeometry(20,160,200,200) 
        self.option2 = QPushButton(self.Optionpage)
        self.option2.setText("(2)\n"+'옵션2: \n'+self.topping[1][0]+'\n가격: '+str(self.topping[1][1])) 
        self.option2.setGeometry(235,160,200,200)
        self.checkbtn2 = QPushButton(self.Optionpage)
        self.checkbtn2.setText('담기') 
        self.checkbtn2.setGeometry(590,780,60,40) 

        self.Pay_label = QLabel(self.Selectpage)
        self.Pay_label.setFrameShape(QFrame.Box)  
        self.Pay_label.setText('결제')
        self.Pay_label.setAlignment(Qt.AlignCenter)
        self.Pay_label.setGeometry(40,160,680,130)
        self.SlcCard = QPushButton(self.Selectpage)
        self.SlcCard.setText('카드')
        self.SlcCard.setGeometry(140,315,225,225)
        self.SlcQR = QPushButton(self.Selectpage)
        self.SlcQR.setText('QR코드')
        self.SlcQR.setGeometry(390,315,225,225) 

        self.Number_label = QLabel(self.Paypage)
        self.Number_label.setFrameShape(QFrame.Box)  
        self.Number_label.setText('')
        self.Number_label.setAlignment(Qt.AlignCenter)
        self.Number_label.setGeometry(40,160,680,130)
        self.Check_label = QLabel(self.Paypage)
        self.Check_label.setFrameShape(QFrame.Box)  
        self.Check_label.setText('')
        self.Check_label.setAlignment(Qt.AlignCenter)
        self.Check_label.setGeometry(40,315,400,400)
        self.Detail_label = QLabel(self.Paypage)
        self.Detail_label.setFrameShape(QFrame.Box)  
        self.Detail_label.setText('QR 코드를 대고 OK 하십시오')
        self.Detail_label.setAlignment(Qt.AlignCenter)
        self.Detail_label.setGeometry(455,315,265,400)
        self.checkbtn3 = QPushButton(self.Paypage)
        self.checkbtn3.setText('OK')
        self.checkbtn3.setGeometry(645,660,60,40)

        self.Number_label2 = QLabel(self.Paypage2)
        self.Number_label2.setFrameShape(QFrame.Box)  
        self.Number_label2.setText('')
        self.Number_label2.setAlignment(Qt.AlignCenter)
        self.Number_label2.setGeometry(40,160,680,130)
        self.Check_label2 = QLabel(self.Paypage2)
        self.Check_label2.setFrameShape(QFrame.Box)  
        self.Check_label2.setText('')
        self.Check_label2.setAlignment(Qt.AlignCenter)
        self.Check_label2.setGeometry(40,315,400,400)
        self.Detail_label2 = QLabel(self.Paypage2)
        self.Detail_label2.setFrameShape(QFrame.Box)  
        self.Detail_label2.setText('확인 하셨다면 OK 하십시오')
        self.Detail_label2.setAlignment(Qt.AlignCenter)
        self.Detail_label2.setGeometry(455,315,265,400)
        self.checkbtn4 = QPushButton(self.Paypage2)
        self.checkbtn4.setText('OK')
        self.checkbtn4.setGeometry(645,655,60,40)
        
        self.receipt_label = QLabel(self.Receiptpage)
        self.receipt_label.setFrameShape(QFrame.Box)  
        self.receipt_label.setText('Check')
        self.receipt_label.setAlignment(Qt.AlignCenter)
        self.receipt_label.setGeometry(40, 120, 680, 560)
        self.gotomain_label = QLabel(self.Receiptpage)
        self.gotomain_label.setFrameShape(QFrame.Box)  
        self.gotomain_label.setText('확인을 누르면 5초 뒤 메인화면으로 돌아갑니다.')
        self.gotomain_label.setAlignment(Qt.AlignCenter)
        self.gotomain_label.setGeometry(40,700,500,120)
        self.checkbtn5 = QPushButton(self.Receiptpage)
        self.checkbtn5.setText('확인')
        self.checkbtn5.setGeometry(555,700,165,120)

        self.btns = [ 
            self.btn1, self.btn2, self.btn3, 
            self.option1, self.option2, self.checkbtn5
        ]
        
        self.exceptbtns = [
            self.Prebtn1, self.Prebtn2, self.Prebtn3,
            self.checkbtn1, self.checkbtn2, self.checkbtn3, self.checkbtn4,
            self.Reset
        ]

        self.Slcbtns = [
            self.SlcCard, self.SlcQR
        ]

        self.Allbtns = [
            self.btns, self.exceptbtns, self.Slcbtns
        ]

        self.BigLabels = [
            self.Pay_label
        ]

        self.MiddleLabels = [
            self.Number_label, 
            self.Number_label2,
            self.Check_label, 
            self.Check_label2,
        ]

        self.SmallLabels = [
            self.Main_check,
            self.Detail_label,
            self.Detail_label2,
            self.receipt_label,
            self.gotomain_label,
        ]

        self.AllLabels = [self.BigLabels, self.MiddleLabels, self.SmallLabels]

        #버튼 폰트 및 색상, 크기 지정 
        for bf in range(len(self.Allbtns)):
            for btnfont in self.Allbtns[bf]:
                self.font = btnfont.font()
                self.font.setFamily('fantasy')
                btnfont.setStyleSheet(
                        "color: blue;"
                        "background-color: #87CEFA;"
                        "border-style: solid;"
                        "border-width: 3px;"
                        "border-color: #1E90FF"
                        )

                if self.Allbtns[bf] == self.Allbtns[0]:
                    self.font.setBold(True)
                    self.font.setPointSize(15)

                if self.Allbtns[bf] == self.Allbtns[1]:
                    self.font.setBold(False)
                    self.font.setPointSize(10)
        
                if self.Allbtns[bf] == self.Allbtns[2]:
                    self.font.setBold(True)
                    self.font.setPointSize(25)

                btnfont.setFont(self.font)

        #라벨 폰트 및 색상, 크기 지정 
        for lf in range(len(self.AllLabels)):
            for labelfont in self.AllLabels[lf]:
                self.font2 = labelfont.font()
                self.font2.setFamily('fantasy')
                labelfont.setStyleSheet(
                        "color: blue;"
                        "background-color: #87CEFA;"
                        "border-style: solid;"
                        "border-width: 3px;"
                        "border-color: #1E90FF"
                        )
                
                if self.AllLabels[lf] == self.AllLabels[0]:
                    self.font2.setBold(True)
                    self.font2.setPointSize(25)

                if self.AllLabels[lf] == self.AllLabels[1]:
                    self.font2.setBold(True)
                    self.font2.setPointSize(17)
                
                if self.AllLabels[lf] == self.AllLabels[2]:
                    self.font2.setBold(True)
                    self.font2.setPointSize(13)
                
                labelfont.setFont(self.font2)
        
        self.stack.setStyleSheet('background:rgb(25,255,200)')
        
        # 내용입력이 완료된 페이지를 QStackedWidget객체에 추가
        self.stack.addWidget(self.Mainpage) 
        self.stack.addWidget(self.Optionpage) 
        self.stack.addWidget(self.Selectpage)
        self.stack.addWidget(self.Paypage)
        self.stack.addWidget(self.Paypage2)
        self.stack.addWidget(self.Receiptpage) 

        self.Reset.clicked.connect(self.ResetAll)
        self.btn1.clicked.connect(self.Menu1)
        self.btn2.clicked.connect(self.Menu2)
        self.btn3.clicked.connect(self.Menu3)
        self.option1.clicked.connect(self.Option1)
        self.option2.clicked.connect(self.Option2)
        self.checkbtn1.clicked.connect(self.GotoSlc)
        self.checkbtn2.clicked.connect(self.PutItIn)
        self.checkbtn3.clicked.connect(self.QRPay)
        self.checkbtn4.clicked.connect(self.Check)
        self.checkbtn5.clicked.connect(self.GotoMain)
        self.Prebtn1.clicked.connect(self.PrePage)
        self.Prebtn2.clicked.connect(self.PrePage)
        self.Prebtn3.clicked.connect(self.PrePage)
        self.SlcQR.clicked.connect(self.QRPay)
        self.stack.setCurrentIndex(0)

    def ResetAll(self):
        """변수 및 페이지 초기화"""
        global barcode_data
        self.nowpg = 0
        self.run = True
        self.topping = [['HOT', 0], ['ICE', 500]]
        self.basket = []
        self.OptL = []
        self.sum_option = 0
        self.exe_last_fnc_Detect = False
        self.select_menu = ""
        self.total_price = 0
        self.MenuText = ''
        self.OptionText = ''
        self.menu = [["아메리카노", 2500], ["카페라떼", 3000], ["에스프레소", 2000]]
        self.MainText = 'Shopping Basket: '
        self.new_text = ''
        self.Main_check.setText('Shopping Basket: 0')
        barcode_data = ''
        self.stack.setCurrentWidget(self.Mainpage)

    def Menu1(self):
        self.nowpg += 1
        self.stack.setCurrentIndex(self.nowpg) 
        self.select_menu = self.menu[0]

    def Menu2(self):
        self.nowpg += 1
        self.stack.setCurrentIndex(self.nowpg) 
        self.select_menu = self.menu[1]

    def Menu3(self):
        self.nowpg += 1
        self.stack.setCurrentIndex(self.nowpg) 
        self.select_menu = self.menu[2]

    def Option1(self):
        print("option1") 
        self.basket = self.topping[0]
        print(self.basket)

    def Option2(self):
        print("option2")
        self.basket = self.topping[1]
        print(self.basket)

    def PutItIn(self):
        self.nowpg = 0
        self.stack.setCurrentWidget(self.Mainpage)
        
        self.MenuText = "\n{}: {}".format(self.select_menu[0], self.select_menu[1])
        self.OptionText = " + {}: {}".format(self.basket[0], self.basket[1])
        self.MainText = self.MainText+self.MenuText+self.OptionText
        self.total_price += int(self.basket[1]) + int(self.select_menu[1])
        self.Main_check.setText(self.MainText)
        
        print("total: "+str(self.select_menu)+str(self.basket))
        print("합계 금액: "+str(self.total_price))
        print(self.MainText)
        self.basket = []


    def GotoSlc(self):
        """선택 페이지로 이동"""
        self.nowpg = 2
        self.new_text = self.MainText.replace('Shopping Basket: ', '')
        self.new_text = "메뉴 선택: \n"+self.new_text+"\n\n합계 금액: "+str(self.total_price)
        self.Check_label.setText(self.new_text)
        self.stack.setCurrentWidget(self.Selectpage)

    def QRPay(self):
        """QR코드 인식 및 내용 표시"""
        global QRrun
        global barcode_data
        global new_QRdata

        new_QRdata = ''
        barcode_data =  str(barcode_data)
        barcode_data = barcode_data.split(sep=',')
        
        if barcode_data == ["['']"]:
            barcode_data = ['']

        new_QRdata = "{}".format(barcode_data[0])

        for d in range(len(barcode_data)-1):
            new_QRdata = new_QRdata+"-"+barcode_data[d+1]

        if self.stack.currentWidget() == self.Selectpage:
            self.nowpg = 3
            self.Number_label.setText("Barcode Number: 0")
            self.stack.setCurrentWidget(self.Paypage)

        if self.stack.currentWidget() == self.Paypage:
            QRrun = True

            if barcode_data != ['']:
                self.nowpg = 4
                self.Check_label2.setText(self.new_text)
                self.Number_label2.setText("Barcode Number: "+new_QRdata)
                self.stack.setCurrentWidget(self.Paypage2)

        print(barcode_data)

    def Check(self):
        if self.stack.currentWidget() == self.Paypage2:
            self.nowpg = 5
        
        self.RecepitBasic_text = ["\n주소:대전광역시 서구 둔산동...",
                        "\n대표:ㅁㅁㅁ             전화:042-1234-1234",
                        "\n=======================================================================================\n\n"]
        self.receipt_text = ""

        for rbtxt in self.RecepitBasic_text:
            self.receipt_text += rbtxt

        self.last_receipt_text = self.receipt_text+self.new_text
        self.receipt_label.setText(self.last_receipt_text)
        self.stack.setCurrentWidget(self.Receiptpage)

    def PrePage(self):
        """이전 페이지로 이동"""
        if self.stack.currentWidget() == self.Receiptpage:
            self.nowpg -= 1
            self.stack.setCurrentIndex(self.nowpg) 

        if self.stack.currentWidget() == self.Selectpage:
            self.nowpg -= 2
            self.stack.setCurrentIndex(self.nowpg) 

        if self.stack.currentWidget() == self.Optionpage:
            self.nowpg -= 1
            self.basket = []
            self.stack.setCurrentIndex(self.nowpg) 

    def GotoMain(self):
        """메인 화면으로 이동"""
        global barcode_data
        global QRrun

        if self.stack.currentWidget() == self.Receiptpage:
            self.exe_last_fnc_Detect = True

            for timer in range(5):
                print(5-timer)
                time.sleep(1)
        
        self.ResetAll()

    def center(self):
        """화면 가운데에 표시"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, QCloseEvent):
        """종료 이벤트"""
        print("Bye")
        QCloseEvent.accept()

    def fnc_Detect(self):
        """제스처 인식에 따른 UI이벤트 실행"""
        if Detect_data == 'ONE':
            if self.stack.currentWidget() == self.Mainpage:
                self.btn1.click()
            
            if self.stack.currentWidget() == self.Optionpage:
                self.option1.click()

        if Detect_data == 'TWO':
            if self.stack.currentWidget() == self.Mainpage:
                self.btn2.click()
            
            if self.stack.currentWidget() == self.Optionpage:
                self.option2.click()
            
            if self.stack.currentWidget() == self.Selectpage:
                self.SlcQR.click()

        if Detect_data == 'FIVE':
            if self.stack.currentWidget() == self.Mainpage:
                self.btn3.click()

            if self.stack.currentWidget() == self.Optionpage:
                self.option3.click()

        if Detect_data == 'OK':
            if self.stack.currentWidget() == self.Mainpage:
                self.checkbtn1.click()
            
            if self.stack.currentWidget() == self.Optionpage:
                self.checkbtn2.click()

            if self.stack.currentWidget() == self.Paypage:
                self.checkbtn3.click()
            
            if self.stack.currentWidget() == self.Paypage2:
                self.checkbtn4.click()

            if self.stack.currentWidget() == self.Receiptpage:
                self.checkbtn5.click()

        if Detect_data == 'GOOD' or Detect_data == 'ZERO':
            if self.stack.currentWidget() == self.Mainpage:
                self.Reset.click()

            if self.stack.currentWidget() == self.Optionpage:
                self.Prebtn1.click()

            if self.stack.currentWidget() == self.Selectpage:
                self.Prebtn2.click()

            if self.stack.currentWidget() == self.Receiptpage:
                self.Prebtn3.click()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = App() #메인 스레드

    thread1 = threading.Thread(target=last_fnc_detect, daemon = True) #스레드 생성, 메인 스레드가 꺼질때 같이 종료
    thread1.start()

    MainWindow.show()
    sys.exit(app.exec_())
    