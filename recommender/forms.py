from django import forms

class BookForm(forms.Form):
    title = forms.CharField(label='Enter a Book Title', max_length=255)
