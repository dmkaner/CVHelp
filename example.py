image_copy = np.copy(image)

class ToTensor(object):

    def __call__(self, sample):
        image = sample

        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    print(predicted_key_pts.shape)
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
#     plt.scatter(predicted_key_pts[:,:, 0], predicted_key_pts[:,:, 1], s=20, marker='.', c='m')

# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:

    # Select the region of interest that is the face in the image
    roi = image_copy[y:y+h, x:x+w]

    ## TODO: Convert the face region from RGB to grayscale
    image_gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
    image_gray = image_gray/255.0
    image_gray = cv2.resize(image_gray,(224,224))

    toTensor = ToTensor()
#     image_gray_ = torch.Tensor(image_gray)
    image_gray_toTensor = toTensor(image_gray)
    image_gray_ = image_gray_toTensor.view(1,1,224,224)

    image_gray_= image_gray_.type(torch.FloatTensor)

    output_pts = net(image_gray_)

    ## TODO: Display each detected face and the corresponding keypoints
    show_all_keypoints(image_gray, output_pts.data.numpy()*50.0+100)
