from django.conf import settings
import tensorflow as tf
from django.shortcuts import redirect, render
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from PIL import Image
import numpy as np
from .forms import MyForm
from .models import MyModel
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.models import Model
from django.http import JsonResponse
import concurrent.futures

class_to_num_dog = {'affenpinscher': 0,
 'afghan_hound': 1,
 'african_hunting_dog': 2,
 'airedale': 3,
 'american_staffordshire_terrier': 4,
 'appenzeller': 5,
 'australian_terrier': 6,
 'basenji': 7,
 'basset': 8,
 'beagle': 9,
 'bedlington_terrier': 10,
 'bernese_mountain_dog': 11,
 'black-and-tan_coonhound': 12,
 'blenheim_spaniel': 13,
 'bloodhound': 14,
 'bluetick': 15,
 'border_collie': 16,
 'border_terrier': 17,
 'borzoi': 18,
 'boston_bull': 19,
 'bouvier_des_flandres': 20,
 'boxer': 21,
 'brabancon_griffon': 22,
 'briard': 23,
 'brittany_spaniel': 24,
 'bull_mastiff': 25,
 'cairn': 26,
 'cardigan': 27,
 'chesapeake_bay_retriever': 28,
 'chihuahua': 29,
 'chow': 30,
 'clumber': 31,
 'cocker_spaniel': 32,
 'collie': 33,
 'curly-coated_retriever': 34,
 'dandie_dinmont': 35,
 'dhole': 36,
 'dingo': 37,
 'doberman': 38,
 'english_foxhound': 39,
 'english_setter': 40,
 'english_springer': 41,
 'entlebucher': 42,
 'eskimo_dog': 43,
 'flat-coated_retriever': 44,
 'french_bulldog': 45,
 'german_shepherd': 46,
 'german_short-haired_pointer': 47,
 'giant_schnauzer': 48,
 'golden_retriever': 49,
 'gordon_setter': 50,
 'great_dane': 51,
 'great_pyrenees': 52,
 'greater_swiss_mountain_dog': 53,
 'groenendael': 54,
 'ibizan_hound': 55,
 'irish_setter': 56,
 'irish_terrier': 57,
 'irish_water_spaniel': 58,
 'irish_wolfhound': 59,
 'italian_greyhound': 60,
 'japanese_spaniel': 61,
 'keeshond': 62,
 'kelpie': 63,
 'kerry_blue_terrier': 64,
 'komondor': 65,
 'kuvasz': 66,
 'labrador_retriever': 67,
 'lakeland_terrier': 68,
 'leonberg': 69,
 'lhasa': 70,
 'malamute': 71,
 'malinois': 72,
 'maltese_dog': 73,
 'mexican_hairless': 74,
 'miniature_pinscher': 75,
 'miniature_poodle': 76,
 'miniature_schnauzer': 77,
 'newfoundland': 78,
 'norfolk_terrier': 79,
 'norwegian_elkhound': 80,
 'norwich_terrier': 81,
 'old_english_sheepdog': 82,
 'otterhound': 83,
 'papillon': 84,
 'pekinese': 85,
 'pembroke': 86,
 'pomeranian': 87,
 'pug': 88,
 'redbone': 89,
 'rhodesian_ridgeback': 90,
 'rottweiler': 91,
 'saint_bernard': 92,
 'saluki': 93,
 'samoyed': 94,
 'schipperke': 95,
 'scotch_terrier': 96,
 'scottish_deerhound': 97,
 'sealyham_terrier': 98,
 'shetland_sheepdog': 99,
 'shih-tzu': 100,
 'siberian_husky': 101,
 'silky_terrier': 102,
 'soft-coated_wheaten_terrier': 103,
 'staffordshire_bullterrier': 104,
 'standard_poodle': 105,
 'standard_schnauzer': 106,
 'sussex_spaniel': 107,
 'tibetan_mastiff': 108,
 'tibetan_terrier': 109,
 'toy_poodle': 110,
 'toy_terrier': 111,
 'vizsla': 112,
 'walker_hound': 113,
 'weimaraner': 114,
 'welsh_springer_spaniel': 115,
 'west_highland_white_terrier': 116,
 'whippet': 117,
 'wire-haired_fox_terrier': 118,
 'yorkshire_terrier': 119}

class_to_num_cat = {'Abyssinian': 0,
 'Bengal': 1,
 'Birman': 2,
 'Bombay': 3,
 'British': 4,
 'Egyptian': 5,
 'Maine': 6,
 'Persian': 7,
 'Ragdoll': 8,
 'Russian': 9,
 'Siamese': 10,
 'Sphynx': 11}

InceptionV3 = InceptionV3(weights='imagenet', include_top=False)
Xception = Xception(weights='imagenet', include_top=False)
InceptionResNetV2 = InceptionResNetV2(weights='imagenet', include_top=False)


def get_features(model, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_maps = feature_extractor.predict(data, batch_size=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

dnn_dog = keras.models.Sequential([
    InputLayer((5632,)),
    Dropout(0.7),
    Dense(120, activation='softmax')
])

dnn_dog.load_weights("Models/dog_final_weights_5632.h5")

dnn_dog.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

dnn_cat = keras.models.Sequential([
    InputLayer((5632,)),
    Dropout(0.7),
    Dense(12, activation='softmax')
])

dnn_cat.load_weights("Models/cat_final_weights_5632.h5")

dnn_cat.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

input_size = (224,224,3)

inception_preprocessor = keras.applications.inception_v3.preprocess_input
xception_preprocessor = keras.applications.xception.preprocess_input
inc_resnet_preprocessor = keras.applications.inception_resnet_v2.preprocess_input


def index(request):
    return render(request, 'Authenticator/index.html')

def predict(request):
    if request.method == 'POST':
        form = MyForm(request.POST, request.FILES)
        if form.is_valid():
            choice = form.cleaned_data['pet']
            image = form.cleaned_data['image']
            image = Image.open(image)
            # MyModel.objects.create(image=image)
            test_data = np.array(image)
            test_data = np.resize(test_data, new_shape=input_size)
            test_data = tf.expand_dims(test_data, 0)
 
            inception_features = get_features(InceptionV3, inception_preprocessor, input_size, test_data)
            xception_features = get_features(Xception, xception_preprocessor, input_size, test_data)
            inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, input_size, test_data)

            test_features = np.concatenate([inception_features,
                                            xception_features,
                                            inc_resnet_features],axis=-1)

            if choice == 'dog' or choice == 'Dog':
                y_pred = dnn_dog.predict(test_features)
            else:
                y_pred = dnn_cat.predict(test_features) 

            top5_indices = np.argsort(y_pred[0])[::-1][:5]

            for idx in top5_indices:
                if choice == 'dog' or choice == 'Dog':
                    class_name = [key for key, value in class_to_num_dog.items() if value == idx][0]
                    messages.success(request, f'Dog Breed: {class_name}')
                    return render(request,'Authenticator/index.html')
                else:
                    class_name = [key for key, value in class_to_num_cat.items() if value == idx][0]
                    messages.success(request, f'Cat Breed: {class_name}')
                    return render(request,'Authenticator/index.html')
        else:
            return JsonResponse({'error' : 'Invalid Form'}) 
    else:   
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
def username_exists(username):
    return User.objects.filter(username=username).exists()

def signup(request):

    if request.method == 'POST':
        uname = request.POST.get('uname')
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('pass')
        confirmPass = request.POST.get('confirmPass')

        if not username_exists(uname):
            user = User.objects.create_user(uname, email, password)
            user.first_name = name

            user.save() 

            messages.success(request, 'Your account has been successfully created! ')
            return redirect('signin')

    return render(request, 'Authenticator/signup.html')

def signin(request):

    if request.method == 'POST':
        uname = request.POST.get('uname')
        password = request.POST.get('pass')

        user = authenticate(request, username=uname, password=password)

        if user != None:
            login(request, user)
            name = {'name' : user.first_name}
            messages.success(request, 'You have succesfully Logged in!')
            return render(request, "Authenticator/index.html", name)

        else:
            messages.error(request, "Incorrect username or password!")
            return redirect('signin')

    return render(request, 'Authenticator/signin.html')


def signout(request):
    logout(request)
    messages.success(request, 'You have logged out successfully!')
    return redirect('index')
