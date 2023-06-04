from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.keras.models import load_model

def predict(request):
    if request.method == 'POST':
        # Load the saved model
        model = load_model('model.hdf5')

        # Get the image from the request
        image = request.FILES['image']

        # Make predictions using the model
        predictions = model.predict(image)

        # Return the predictions as a response
        return HttpResponse(predictions)
    else:
        # Render the form to upload an image
        return render(request, 'predict.html')
