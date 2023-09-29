from django import forms

class MyForm(forms.Form):
    
    CHOICES = [('dog', 'Dog'), ('cat', 'Cat')]
    
    pet = forms.ChoiceField(choices=CHOICES, widget=forms.RadioSelect)
    image = forms.ImageField()
