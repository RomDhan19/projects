import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the image path using os.path for robust path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'chien.jpg')

# Load the color image
color_image = cv2.imread(image_path)

# Check if the image was loaded successfully
if color_image is None:
    print(f'Error: Could not load image from {image_path}')
    print(f'Current working directory: {os.getcwd()}')
    print(f'Full image path: {image_path}')
else:
    print(f'Successfully loaded color image with shape: {color_image.shape}')

    # Separate the color channels (OpenCV loads as BGR by default)
    b_channel, g_channel, r_channel = cv2.split(color_image)

    print('Successfully separated color channels.')

    # Create a grayscale image
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    print(f'Successfully created grayscale image with shape: {gray_image.shape}')

    # --- Display Images ---

    # Display individual channels
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Color')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(r_channel, cmap='gray')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(g_channel, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(b_channel, cmap='gray')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Display paired combinations (as pseudo-color images)
    # For display purposes, we can stack the desired channels with a zero channel
    # to create an image that matplotlib can display.

    # RG combination (Red and Green channels, Blue channel set to 0)
    rg_image = cv2.merge([np.zeros_like(b_channel), g_channel, r_channel])
    # GB combination (Green and Blue channels, Red channel set to 0)
    gb_image = cv2.merge([b_channel, g_channel, np.zeros_like(r_channel)])
    # RB combination (Red and Blue channels, Green channel set to 0)
    rb_image = cv2.merge([b_channel, np.zeros_like(g_channel), r_channel])

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(rg_image, cv2.COLOR_BGR2RGB))
    plt.title('RG Combination')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(gb_image, cv2.COLOR_BGR2RGB))
    plt.title('GB Combination')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(rb_image, cv2.COLOR_BGR2RGB))
    plt.title('RB Combination')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Display grayscale image
    plt.figure(figsize=(5, 5))
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()

    # --- Plotting Channel Frequency for Selected Rows ---

    # Select a few rows to plot (adjust indices based on image height)
    # Let's choose rows at 1/4, 1/2, and 3/4 of the image height.
    height, width, _ = color_image.shape
    selected_rows = [height // 4, height // 2, (height * 3) // 4]

    print(f"Plotting channel frequencies for rows: {selected_rows}")

    # Plot for Color Image Channels (R, G, B)
    plt.figure(figsize=(15, 10))
    plt.suptitle('Color Channel Intensities Across Selected Rows')

    for i, row_index in enumerate(selected_rows):
        plt.subplot(len(selected_rows), 1, i + 1)
        plt.plot(r_channel[row_index, :], color='red', label='Red')
        plt.plot(g_channel[row_index, :], color='green', label='Green')
        plt.plot(b_channel[row_index, :], color='blue', label='Blue') # OpenCV is BGR, so this is the blue channel
        plt.title(f'Row {row_index}')
        plt.xlabel('Column Index')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()

    # Plot for Grayscale Image
    plt.figure(figsize=(15, 10))
    plt.suptitle('Grayscale Intensities Across Selected Rows')

    for i, row_index in enumerate(selected_rows):
        plt.subplot(len(selected_rows), 1, i + 1)
        plt.plot(gray_image[row_index, :], color='gray', label='Grayscale')
        plt.title(f'Row {row_index}')
        plt.xlabel('Column Index')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()

    # --- Image Filtering (Convolution) ---

    # Generic convolution function with zero-padding
    def apply_convolution(image, kernel):
        # Get kernel dimensions
        k_height, k_width = kernel.shape

        # Get image dimensions
        i_height, i_width = image.shape

        # Calculate padding amounts
        pad_y = k_height // 2
        pad_x = k_width // 2

        # Pad the image with zeros
        padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

        # Create output image
        output_image = np.zeros_like(image, dtype=np.float64)

        # Perform convolution
        for y in range(i_height):
            for x in range(i_width):
                # Extract the region of interest (ROI)
                roi = padded_image[y:y + k_height, x:x + k_width]

                # Perform element-wise multiplication and sum
                output_image[y, x] = np.sum(roi * kernel)

        # Clip values to be within 0-255 and convert to uint8
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)

        return output_image

    # --- Filter Implementations ---

    # (a) Thresholding filter
    def apply_thresholding(image, threshold):
        # User adjustable parameter: threshold
        _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return thresholded_image

    # (b) Median filter
    def apply_median_filter(image, kernel_size):
        # User adjustable parameter: kernel_size (should be odd)
        median_filtered_image = cv2.medianBlur(image, kernel_size)
        return median_filtered_image

    # (c) Gaussian filter
    def apply_gaussian_filter(image, kernel_size, sigma_x):
        # User adjustable parameters: kernel_size (should be odd), sigma_x
        gaussian_filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)
        return gaussian_filtered_image

    # (d) Linear averaging filter (box blur)
    def apply_box_blur(image, kernel_size):
        # User adjustable parameter: kernel_size
        # Create a box kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        # Apply convolution
        box_blurred_image = apply_convolution(image, kernel)
        return box_blurred_image

    # (e) Sobel filter
    def apply_sobel_filter(image, ksize=3):
        # User adjustable parameter: ksize (Sobel kernel size, must be 1, 3, 5, or 7)

        # Apply Sobel filter in the x direction
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        # Apply Sobel filter in the y direction
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

        # Compute the gradient magnitude
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Scale and convert to uint8
        sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

        return sobel_magnitude

    # --- Apply and Display Filters (Example) ---

    # Apply Thresholding filter (example threshold: 127)
    threshold_value = 127 # Example parameter
    thresholded_image = apply_thresholding(gray_image, threshold_value)

    # Apply Median filter (example kernel size: 5)
    median_kernel_size = 5 # Example parameter (must be odd)
    median_filtered_image = apply_median_filter(gray_image, median_kernel_size)

    # Apply Gaussian filter (example kernel size: 5, sigma_x: 0)
    gaussian_kernel_size = 5 # Example parameter (must be odd)
    gaussian_sigma_x = 0 # Example parameter
    gaussian_filtered_image = apply_gaussian_filter(gray_image, gaussian_kernel_size, gaussian_sigma_x)

    # Apply Box Blur filter (example kernel size: 5)
    box_blur_kernel_size = 5 # Example parameter
    box_blurred_image = apply_box_blur(gray_image, box_blur_kernel_size)

    # Apply Sobel filter (example kernel size: 3)
    sobel_ksize = 3 # Example parameter (must be 1, 3, 5, or 7)
    sobel_filtered_image = apply_sobel_filter(gray_image, sobel_ksize)

    # Display filtered images
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title(f'Thresholded (>{threshold_value})')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(median_filtered_image, cmap='gray')
    plt.title(f'Median Filter (kernel={median_kernel_size})')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(gaussian_filtered_image, cmap='gray')
    plt.title(f'Gaussian Filter (kernel={gaussian_kernel_size}, sigma={gaussian_sigma_x})')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(box_blurred_image, cmap='gray')
    plt.title(f'Box Blur (kernel={box_blur_kernel_size})')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(sobel_filtered_image, cmap='gray')
    plt.title(f'Sobel Filter (ksize={sobel_ksize})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Image Enlargement using Fourier Series ---

    def calculate_fourier_coefficients(image, m, n):
        # Get image dimensions
        l1, l2 = image.shape # l1 is height, l2 is width

        a_coefficients = {}

        # Calculate up to max_m and max_n that make sense for the original image size and desired m, n.
        # We need coefficients for j from 0 to m and k from 0 to n for the symmetry optimization.
        # The maximum frequency component we can capture from an image of size L is related to L/2.
        # Let's calculate coefficients up to m and n, but cap at image dimensions for the discrete sum.
        max_m_calc = min(m, l1 - 1)
        max_n_calc = min(n, l2 - 1)

        print(f"Calculating coefficients for j from 0 to {max_m_calc} and k from 0 to {max_n_calc}")

        for j_c in range(max_m_calc + 1):
            for k_c in range(max_n_calc + 1):
                sum_val = 0.0
                # Iterate through image pixels (row by column)
                for y_orig_idx in range(image.shape[0]): # Iterate through rows (height)
                    for x_orig_idx in range(image.shape[1]): # Iterate through columns (width)
                        # Explicit check to prevent index error (should not be necessary with range())
                        if y_orig_idx < image.shape[0] and x_orig_idx < image.shape[1]:
                            # Adjust cosine arguments based on the assumption that x in formula maps to column (l2) and y maps to row (l1)
                            sum_val += image[y_orig_idx, x_orig_idx] * np.cos(np.pi * j_c * x_orig_idx / l2) * np.cos(np.pi * k_c * y_orig_idx / l1)
                        else:
                             print(f"Warning: Unexpected index out of bounds - y_orig_idx: {y_orig_idx}, x_orig_idx: {x_orig_idx}, image.shape: {image.shape}")

                a_coefficients[(j_c, k_c)] = sum_val / (l1 * l2)

        return a_coefficients, l1, l2

    def reconstruct_image_fourier(a_coefficients, original_l1, original_l2, m, n, scale_factor):
        # Calculate new image dimensions
        new_l1 = int(original_l1 * scale_factor)
        new_l2 = int(original_l2 * scale_factor)

        # Create output image
        reconstructed_image = np.zeros((new_l2, new_l1), dtype=np.float64)

        # Get the max calculated indices from the coefficient dictionary
        if a_coefficients:
             max_calc_m = max(key[0] for key in a_coefficients.keys())
             max_calc_n = max(key[1] for key in a_coefficients.keys())
        else:
             max_calc_m = -1
             max_calc_n = -1

        for x_new in range(new_l1):
            for y_new in range(new_l2):
                # Map new image coordinates to original image scale for cosine arguments
                x_scaled = x_new / scale_factor
                y_scaled = y_new / scale_factor

                smn_val = 0.0
                for j in range(-m, m + 1):
                    for k in range(-n, n + 1):
                        abs_j = abs(j)
                        abs_k = abs(k)

                        if abs_j <= max_calc_m and abs_k <= max_calc_n:
                            coeff = a_coefficients.get((abs_j, abs_k), 0.0)
                            # Adjust cosine arguments based on the assumption that x in formula maps to column (original_l2) and y maps to row (original_l1)
                            smn_val += coeff * np.cos(np.pi * j * x_scaled / original_l2) * np.cos(np.pi * k * y_scaled / original_l1)

                reconstructed_image[y_new, x_new] = smn_val

        # Clip values to be within 0-255 and convert to uint8
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        return reconstructed_image

    # --- Apply Image Enlargement (Example) ---

    # Define enlargement parameters
    enlargement_factor = 2 # Example: enlarge by 2 times
    m_sum = 30 # Reduced from 20 to 15 for better performance
    n_sum = 30 # Reduced from 30 to 15 for better performance

    print(f"Calculating Fourier coefficients for m up to {m_sum}, n up to {n_sum}")
    # Calculate coefficients based on the original grayscale image and the desired summation limits m and n
    a_coeffs, orig_l1, orig_l2 = calculate_fourier_coefficients(gray_image, m_sum, n_sum)

    print(f"Reconstructing image with enlargement factor {enlargement_factor}, m={m_sum}, n={n_sum}")
    # Reconstruct the image using the calculated coefficients and the desired enlargement factor
    enlarged_image_fourier = reconstruct_image_fourier(a_coeffs, orig_l1, orig_l2, m_sum, n_sum, enlargement_factor)

    # --- Compare with standard interpolation (e.g., bilinear) ---
    new_width = int(width * enlargement_factor)
    new_height = int(height * enlargement_factor)
    enlarged_image_bilinear = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # --- Display Enlarged Images ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(enlarged_image_fourier, cmap='gray')
    plt.title(f'Enlarged (Fourier, m={m_sum}, n={n_sum})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enlarged_image_bilinear, cmap='gray')
    plt.title(f'Enlarged (Bilinear Interpolation)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Step 9: Fourier Spectrum ---

    print("Calculating and displaying Fourier spectrum...")

    # Calculate the 2D DFT of the grayscale image
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center
    dft_shift = np.fft.fftshift(dft)

    # Calculate the magnitude spectrum (logarithmic scale for better visualization)
    # Magnitude = sqrt(real**2 + imaginary**2)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Display the magnitude spectrum
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([]), plt.yticks([]) # Hide tick values
    plt.show()

    # --- Step 10: High-pass and Low-pass Filtering in Fourier Domain ---

    print("Applying High-pass and Low-pass filters in Fourier domain...")

    # Create masks for filtering
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2 # Center of the spectrum

    # Low-pass filter (Ideal Lowpass Filter - simple example)
    # Create a mask with a radius. Frequencies inside the radius are kept, outside are removed.
    # User adjustable parameter: lowpass_radius
    lowpass_radius = 30 # Example parameter (adjust as needed)
    lowpass_mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(lowpass_mask, (ccol, crow), lowpass_radius, (1, 1), -1)

    # Apply the low-pass mask to the shifted spectrum
    fourier_lowpass = dft_shift * lowpass_mask

    # Inverse DFT to get the filtered image
    fourier_lowpass_ishift = np.fft.ifftshift(fourier_lowpass)
    image_lowpass = cv2.idft(fourier_lowpass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    image_lowpass = np.clip(image_lowpass, 0, 255).astype(np.uint8)

    # High-pass filter (Ideal Highpass Filter - simple example)
    # Create a mask. Frequencies outside the radius are kept, inside are removed.
    # This is the inverse of the low-pass mask.
    # User adjustable parameter: highpass_radius (can be the same as lowpass_radius or different)
    highpass_radius = 30 # Example parameter (adjust as needed)
    highpass_mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(highpass_mask, (ccol, crow), highpass_radius, (0, 0), -1) # Draw a black circle (0) on a white background (1)

    # Apply the high-pass mask to the shifted spectrum
    fourier_highpass = dft_shift * highpass_mask

    # Inverse DFT to get the filtered image
    fourier_highpass_ishift = np.fft.ifftshift(fourier_highpass)
    image_highpass = cv2.idft(fourier_highpass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    image_highpass = np.clip(image_highpass, 0, 255).astype(np.uint8)

    # Display the filtered images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_lowpass, cmap='gray')
    plt.title(f'Low-pass Filtered (radius={lowpass_radius})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_highpass, cmap='gray')
    plt.title(f'High-pass Filtered (radius={highpass_radius})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


