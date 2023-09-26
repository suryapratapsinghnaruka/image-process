# import cv2
# import numpy as np
# from flask import Flask, request, jsonify, send_from_directory
# import os
# import io
# from PIL import Image
# from datetime import datetime
# from fpdf import FPDF
# from io import BytesIO
# from datetime import datetime
# from PyPDF2 import PdfFileWriter, PdfFileReader

# app = Flask(__name__)

# # Set the upload folder
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure the upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/scan', methods=['POST'])
# def scan():
#     # Get image files from the request
#     files = request.files.getlist('image')

#     # List to store image paths
#     image_paths = []

#     for file in files:
#         # Read image file into numpy array
#         img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

#         pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL

#         # Convert the PIL Image to grayscale
#         gray_pil = pil_img.convert('L')

#         # Convert the grayscale PIL Image to numpy array
#         gray = np.array(gray_pil)

#         # Apply threshold to create a binary image
#         _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

#         # Find contours in the binary image
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Find the largest contour (which is likely to be the document)
#         largest_contour = max(contours, key=cv2.contourArea)

#         # Find the convex hull of the largest contour
#         hull = cv2.convexHull(largest_contour)

#         # Approximate the polygonal curve of the convex hull with a simpler curve
#         epsilon = 0.05 * cv2.arcLength(hull, True)
#         approx = cv2.approxPolyDP(hull, epsilon, True)

#         # Ensure that the approximated curve has four vertices
#         if len(approx) != 4:
#             print("Error: The document does not have four corners.")
#         else:
#             # Find the corners of the approximated curve
#             approx = np.squeeze(approx)
#             corners = np.zeros((4, 2), dtype=np.float32)
#             corners[0] = approx[np.argmin(np.sum(approx, axis=1))]
#             corners[2] = approx[np.argmax(np.sum(approx, axis=1))]
#             corners[1] = approx[np.argmin(np.diff(approx, axis=1))]
#             corners[3] = approx[np.argmax(np.diff(approx, axis=1))]

#             # Compute the perspective transform matrix and apply it to the original image
#             height = np.sqrt((corners[2][0] - corners[3][0])**2 + (corners[2][1] - corners[3][1])**2)
#             width = np.sqrt((corners[1][0] - corners[2][0])**2 + (corners[1][1] - corners[2][1])**2)
#             dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
#             M = cv2.getPerspectiveTransform(corners, dst_pts)
#             warped = cv2.warpPerspective(img, M, (int(width), int(height)))

#             warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

#             gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

#             # Estimate the blur kernel (assuming Gaussian blur)
#             kernel_size = 5
#             blur_kernel = cv2.getGaussianKernel(kernel_size, 0)
#             blur_kernel = blur_kernel @ blur_kernel.T

#             # Perform Wiener deconvolution to recover the sharp image
#             psf = np.fft.fft2(blur_kernel, gray.shape)
#             gray_fft = np.fft.fft2(gray)
#             gray_deconv = np.real(np.fft.ifft2(gray_fft / (psf + 1e-8)))
#             gray_deconv = np.uint8(np.clip(gray_deconv, 0, 255))

#             # Perform unsharp masking to enhance the sharpness of the image
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
#             unsharp_mask = np.uint8(np.clip(unsharp_mask, 0, 255))

#             # Save the processed image to a file
#             now = datetime.now()
#             save_path = os.path.join(UPLOAD_FOLDER, f'document_{now.strftime("%Y%m%d_%H%M%S")}.jpg')
#             cv2.imwrite(save_path, unsharp_mask)
#             image_paths.append(save_path)

#     # Generate a PDF document with saved images
#     pdf_path = os.path.join(UPLOAD_FOLDER, f'document_{now.strftime("%Y%m%d_%H%M%S")}.pdf')
#     pdf = FPDF()
#     for image_path in image_paths:
#         pdf.add_page()
#         pdf.image(image_path, 0, 0, pdf.w, pdf.h)
#     pdf.output(pdf_path, 'F')

#     # Provide the base URL for the PDF file
#     pdf_url = f'http://{request.host}/{pdf_path}'

#     return jsonify({'path': pdf_url})

# @app.route('/pdf', methods=['GET'])
# def get_pdf():
#     # Get path to saved PDF document from request
#     pdf_path = request.args.get('path')

#     if pdf_path is None:
#         # Return error response if path is missing
#         return {'error': 'Path parameter is missing.'}, 400

#     # Get directory of saved PDF document
#     directory = os.path.dirname(pdf_path)

#     # Return PDF document in response
#     return send_from_directory(directory, os.path.basename(pdf_path), as_attachment=True)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))



import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import os
import io
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from PyPDF2 import PdfFileWriter, PdfFileReader
import PyPDF2
from PIL import ImageOps


app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_images_from_pdf(pdf_file):
    pdf_images = []
    with BytesIO(pdf_file.read()) as pdf_buffer:
        pdf_reader = PyPDF2.PdfReader(pdf_buffer)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            xObject = page['/Resources']['/XObject'].get_object()
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    img = xObject[obj]
                    img_data = img.get_data()
                    img_format = img['/Filter'][1:]  # Remove the leading '/'
                    img_bytes = io.BytesIO(img_data)

                    # Open the image using PIL and add it to the list
                    pil_img = Image.open(img_bytes)
                    pil_img = ImageOps.grayscale(pil_img)
                    pdf_images.append(pil_img)

    return pdf_images

@app.route('/scan', methods=['POST'])
def scan():
    # Get all uploaded files
    files = request.files.getlist('image')
    pdf_files = request.files.getlist('pdf')

    # List to store image paths
    image_paths = []

    for file in files:
        # Read image file into a numpy array
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL

        # Convert the PIL Image to grayscale
        gray_pil = pil_img.convert('L')

        # Convert the grayscale PIL Image to a numpy array
        gray = np.array(gray_pil)

        # Apply threshold to create a binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (which is likely to be the document)
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the convex hull of the largest contour
        hull = cv2.convexHull(largest_contour)

        # Approximate the polygonal curve of the convex hull with a simpler curve
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # Ensure that the approximated curve has four vertices
        if len(approx) != 4:
            print("Error: The document does not have four corners.")
        else:
            # Find the corners of the approximated curve
            approx = np.squeeze(approx)
            corners = np.zeros((4, 2), dtype=np.float32)
            corners[0] = approx[np.argmin(np.sum(approx, axis=1))]
            corners[2] = approx[np.argmax(np.sum(approx, axis=1))]
            corners[1] = approx[np.argmin(np.diff(approx, axis=1))]
            corners[3] = approx[np.argmax(np.diff(approx, axis=1))]

            # Compute the perspective transform matrix and apply it to the original image
            height = np.sqrt((corners[2][0] - corners[3][0])**2 + (corners[2][1] - corners[3][1])**2)
            width = np.sqrt((corners[1][0] - corners[2][0])**2 + (corners[1][1] - corners[2][1])**2)
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(corners, dst_pts)
            warped = cv2.warpPerspective(img, M, (int(width), int(height)))

            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # Estimate the blur kernel (assuming Gaussian blur)
            kernel_size = 5
            blur_kernel = cv2.getGaussianKernel(kernel_size, 0)
            blur_kernel = blur_kernel @ blur_kernel.T

            # Perform Wiener deconvolution to recover the sharp image
            psf = np.fft.fft2(blur_kernel, gray.shape)
            gray_fft = np.fft.fft2(gray)
            gray_deconv = np.real(np.fft.ifft2(gray_fft / (psf + 1e-8)))
            gray_deconv = np.uint8(np.clip(gray_deconv, 0, 255))

            # Perform unsharp masking to enhance the sharpness of the image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            unsharp_mask = np.uint8(np.clip(unsharp_mask, 0, 255))

            # Save the processed image to a file
            now = datetime.now()
            save_path = os.path.join(UPLOAD_FOLDER, f'document_{now.strftime("%Y%m%d_%H%M%S")}.jpg')
            cv2.imwrite(save_path, unsharp_mask)
            image_paths.append(save_path)

    # Process PDF files if any
    for pdf_file in pdf_files:
        # Extract images from the PDF file
        pdf_images = extract_images_from_pdf(pdf_file)

        # Save the extracted images and get their paths
        for idx, pdf_image in enumerate(pdf_images):
            now = datetime.now()
            save_path = os.path.join(UPLOAD_FOLDER, f'pdf_image_{now.strftime("%Y%m%d_%H%M%S")}_{idx}.jpg')
            pdf_image.save(save_path)
            image_paths.append(save_path)

    # Generate a PDF document with all saved images
    pdf_path = os.path.join(UPLOAD_FOLDER, f'document_{now.strftime("%Y%m%d_%H%M%S")}.pdf')
    pdf = FPDF()
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, 0, 0, pdf.w, pdf.h)
    pdf.output(pdf_path, 'F')

    # Provide the base URL for the PDF file
    pdf_url = f'http://{request.host}/{pdf_path}'

    return jsonify({'path': pdf_url})

@app.route('/pdf', methods=['GET'])
def get_pdf():
    # Get path to saved PDF document from request
    pdf_path = request.args.get('path')

    if pdf_path is None:
        # Return error response if path is missing
        return {'error': 'Path parameter is missing.'}, 400

    # Get directory of saved PDF document
    directory = os.path.dirname(pdf_path)

    # Return PDF document in response
    return send_from_directory(directory, os.path.basename(pdf_path), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
