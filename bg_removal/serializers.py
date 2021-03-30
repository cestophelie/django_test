from rest_framework import serializers
from .models import Post
from .models import Accounts


class PostSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(use_url=True)

    class Meta:
        model = Post
        # fields = ['title']  # 이렇게 단일 필드인 경우는 왜인지 모르겠는데 튜플을 인식하지 않는다. 리스트로 보낸다.
        fields = ('title', 'category', 'image')


class AccountsSerializer(serializers.ModelSerializer):

    class Meta:
        model = Accounts
        fields = ('identify', 'password')
