from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response

from .serializers import PostSerializer
from .models import Post

# 이 부분 추가
from .serializers import AccountsSerializer
from .models import Accounts
# ----

from django.core.files.storage import FileSystemStorage

# for deep learning model
import os
from tensorflow.keras.preprocessing.image import load_img
import threading
import random
import string


# Create your views here.
# 이 부분 추가
class AccountsViewset(viewsets.ModelViewSet):
    queryset = Accounts.objects.all()
    serializer_class = AccountsSerializer


class PostViewset(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

    def create(self, request):  # Here is the new update comes <<<<
        post_data = request.data
        fileObj = request.FILES['image']

        print('-----------')
        print('POST DATA : ' + str(post_data))
        print('FILE OBJECT : ' + str(request.FILES['image']))
        file_name = str(request.FILES['image'])
        file_name = file_name.split('.')
        # 여기서 file_name 은 안드로이드에서 들어온 항상 같은 file_name == "hihi.jpeg"를 split한 "hihi"

        # file_url = file_name[0] + '_.png'
        # 랜덤한 값으로 넣어주기 위해서 file_url 값을 정한다.
        letters = string.ascii_lowercase
        file_url = ''.join(random.choice(letters) for i in range(5)) + '.png'
        # file_name = file_url
        print('FILE URL : ' + str(file_url))

        # images 에 이름은 항상 hihi 로 들어가도록
        file_name = file_name[0]
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        print('filePathName is ' + str(filePathName))
        testimage = '.' + filePathName  # media에 저장된 이미지 이름인데..
        print('testimage is ' + str(testimage))

        img = load_img(testimage)
        directory = os.path.join(os.getcwd(), 'bg_removal/images' + os.sep)  # + os.sep 이거 붙여줘야함
        img.save(directory + file_name + '.jpg')  # image 폴더에 저장

        # DB 테이블에 직접 값 넣어주기
        db_file_url = file_url
        Subs = Post.objects.create(title=file_name, image=db_file_url)
        Subs.save()

        thread = threading.Thread(target=self.hiya, args=(file_url,))
        thread.daemon = True
        thread.start()

        return Response(data='heyhey')
        # return Response(data=str(testimage))

    def hiya(self, file_url):
        command = 'python C:/Users/sewon/django_test/mytestsite/bg_removal/u2net_test.py ' + str(file_url)
        os.system(command)
    # 이 부분은 multithreading 으로 처리해야겠다.
    # 근데 일단 create메소드 안에 들어가는 순간 mysql에는 안 들어가니까 create 메소드 안에 뭔가를 추가해줘야 할 것 같다.
