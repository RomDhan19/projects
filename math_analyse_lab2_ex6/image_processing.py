import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Определение пути к изображению с использованием os.path для надежной обработки путей
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'chien.jpg')

# Загрузка цветного изображения
color_image = cv2.imread(image_path)

# Проверка успешной загрузки изображения
if color_image is None:
    print(f'Error: Could not load image from {image_path}')
    print(f'Current working directory: {os.getcwd()}')
    print(f'Full image path: {image_path}')
else:
    print(f'Successfully loaded color image with shape: {color_image.shape}')

    # Разделение цветовых каналов (OpenCV загружает как BGR по умолчанию)
    b_channel, g_channel, r_channel = cv2.split(color_image)

    print('Successfully separated color channels.')

    # Создание изображения в оттенках серого
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    print(f'Successfully created grayscale image with shape: {gray_image.shape}')

    # --- Отображение изображений ---

    # Отображение отдельных каналов
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

    # Отображение парных комбинаций (как псевдоцветные изображения)
    # Для целей отображения мы можем объединить нужные каналы с нулевым каналом
    # для создания изображения, которое matplotlib может отобразить.

    # RG комбинация (каналы Red и Green, канал Blue установлен в 0)
    rg_image = cv2.merge([np.zeros_like(b_channel), g_channel, r_channel])
    # GB комбинация (каналы Green и Blue, канал Red установлен в 0)
    gb_image = cv2.merge([b_channel, g_channel, np.zeros_like(r_channel)])
    # RB комбинация (каналы Red и Blue, канал Green установлен в 0)
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

    # Отображение изображения в оттенках серого
    plt.figure(figsize=(5, 5))
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()

    # --- Построение графика частоты каналов для выбранных строк ---

    # Выбор нескольких строк для построения графика (настройка индексов в зависимости от высоты изображения)
    # Выберем строки на 1/4, 1/2 и 3/4 высоты изображения.
    height, width, _ = color_image.shape
    selected_rows = [height // 4, height // 2, (height * 3) // 4]

    print(f"Plotting channel frequencies for rows: {selected_rows}")

    # График для цветовых каналов изображения (R, G, B)
    plt.figure(figsize=(15, 10))
    plt.suptitle('Color Channel Intensities Across Selected Rows')

    for i, row_index in enumerate(selected_rows):
        plt.subplot(len(selected_rows), 1, i + 1)
        plt.plot(r_channel[row_index, :], color='red', label='Red')
        plt.plot(g_channel[row_index, :], color='green', label='Green')
        plt.plot(b_channel[row_index, :], color='blue', label='Blue') # OpenCV использует BGR, поэтому это синий канал
        plt.title(f'Row {row_index}')
        plt.xlabel('Column Index')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Настройка макета для предотвращения наложения заголовка
    plt.show()

    # График для изображения в оттенках серого
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Настройка макета
    plt.show()

    # --- Фильтрация изображения (свертка) ---

    # Общая функция свертки с нулевым заполнением
    def apply_convolution(image, kernel):
        # Получение размеров ядра
        k_height, k_width = kernel.shape

        # Получение размеров изображения
        i_height, i_width = image.shape

        # Расчет количества заполнения
        pad_y = k_height // 2
        pad_x = k_width // 2

        # Заполнение изображения нулями
        padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

        # Создание выходного изображения
        output_image = np.zeros_like(image, dtype=np.float64)

        # Выполнение свертки
        for y in range(i_height):
            for x in range(i_width):
                # Извлечение области интереса (ROI)
                roi = padded_image[y:y + k_height, x:x + k_width]

                # Выполнение поэлементного умножения и суммирования
                output_image[y, x] = np.sum(roi * kernel)

        # Ограничение значений в диапазоне 0-255 и преобразование в uint8
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)

        return output_image

    # --- Реализации фильтров ---

    # (a) Пороговый фильтр
    def apply_thresholding(image, threshold):
        # Настраиваемый пользователем параметр: порог
        _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return thresholded_image

    # (b) Медианный фильтр
    def apply_median_filter(image, kernel_size):
        # Настраиваемый пользователем параметр: размер ядра (должен быть нечетным)
        median_filtered_image = cv2.medianBlur(image, kernel_size)
        return median_filtered_image

    # (c) Гауссовский фильтр
    def apply_gaussian_filter(image, kernel_size, sigma_x):
        # Настраиваемые пользователем параметры: размер ядра (должен быть нечетным), sigma_x
        gaussian_filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)
        return gaussian_filtered_image

    # (d) Линейный усредняющий фильтр (box blur)
    def apply_box_blur(image, kernel_size):
        # Настраиваемый пользователем параметр: размер ядра
        # Создание box-ядра
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        # Применение свертки
        box_blurred_image = apply_convolution(image, kernel)
        return box_blurred_image

    # (e) Фильтр Собеля
    def apply_sobel_filter(image, ksize=3):
        # Настраиваемый пользователем параметр: ksize (размер ядра Собеля, должен быть 1, 3, 5 или 7)

        # Применение фильтра Собеля в направлении x
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        # Применение фильтра Собеля в направлении y
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

        # Вычисление величины градиента
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Масштабирование и преобразование в uint8
        sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

        return sobel_magnitude

    # --- Применение и отображение фильтров (пример) ---

    # Применение порогового фильтра (пример порога: 127)
    threshold_value = 127 # Пример параметра
    thresholded_image = apply_thresholding(gray_image, threshold_value)

    # Применение медианного фильтра (пример размера ядра: 5)
    median_kernel_size = 5 # Пример параметра (должен быть нечетным)
    median_filtered_image = apply_median_filter(gray_image, median_kernel_size)

    # Применение гауссовского фильтра (пример размера ядра: 5, sigma_x: 0)
    gaussian_kernel_size = 5 # Пример параметра (должен быть нечетным)
    gaussian_sigma_x = 0 # Пример параметра
    gaussian_filtered_image = apply_gaussian_filter(gray_image, gaussian_kernel_size, gaussian_sigma_x)

    # Применение фильтра Box Blur (пример размера ядра: 5)
    box_blur_kernel_size = 5 # Пример параметра
    box_blurred_image = apply_box_blur(gray_image, box_blur_kernel_size)

    # Применение фильтра Собеля (пример размера ядра: 3)
    sobel_ksize = 3 # Пример параметра (должен быть 1, 3, 5 или 7)
    sobel_filtered_image = apply_sobel_filter(gray_image, sobel_ksize)

    # Отображение отфильтрованных изображений
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

    # --- Увеличение изображения с использованием рядов Фурье ---

    def calculate_fourier_coefficients(image, m, n):
        # Получение размеров изображения
        l1, l2 = image.shape # l1 - высота, l2 - ширина

        a_coefficients = {}

        # Вычисление до max_m и max_n, которые имеют смысл для исходного размера изображения и желаемых m, n.
        # Нам нужны коэффициенты для j от 0 до m и k от 0 до n для оптимизации симметрии.
        # Максимальная частотная составляющая, которую мы можем захватить из изображения размера L, связана с L/2.
        # Давайте вычислим коэффициенты до m и n, но ограничим размером изображения для дискретной суммы.
        max_m_calc = min(m, l1 - 1)
        max_n_calc = min(n, l2 - 1)

        print(f"Calculating coefficients for j from 0 to {max_m_calc} and k from 0 to {max_n_calc}")

        for j_c in range(max_m_calc + 1):
            for k_c in range(max_n_calc + 1):
                sum_val = 0.0
                # Итерация по пикселям изображения (строка за строкой)
                for y_orig_idx in range(image.shape[0]): # Итерация по строкам (высота)
                    for x_orig_idx in range(image.shape[1]): # Итерация по столбцам (ширина)
                        # Явная проверка для предотвращения ошибки индекса (не должна быть необходимой с range())
                        if y_orig_idx < image.shape[0] and x_orig_idx < image.shape[1]:
                            # Настройка аргументов косинуса на основе предположения, что x в формуле соответствует столбцу (l2), а y соответствует строке (l1)
                            sum_val += image[y_orig_idx, x_orig_idx] * np.cos(np.pi * j_c * x_orig_idx / l2) * np.cos(np.pi * k_c * y_orig_idx / l1)
                        else:
                             print(f"Warning: Unexpected index out of bounds - y_orig_idx: {y_orig_idx}, x_orig_idx: {x_orig_idx}, image.shape: {image.shape}")

                a_coefficients[(j_c, k_c)] = sum_val / (l1 * l2)

        return a_coefficients, l1, l2

    def reconstruct_image_fourier(a_coefficients, original_l1, original_l2, m, n, scale_factor):
        # Вычисление новых размеров изображения
        new_l1 = int(original_l1 * scale_factor)
        new_l2 = int(original_l2 * scale_factor)

        # Создание выходного изображения
        reconstructed_image = np.zeros((new_l2, new_l1), dtype=np.float64)

        # Получение максимальных вычисленных индексов из словаря коэффициентов
        if a_coefficients:
             max_calc_m = max(key[0] for key in a_coefficients.keys())
             max_calc_n = max(key[1] for key in a_coefficients.keys())
        else:
             max_calc_m = -1
             max_calc_n = -1

        for x_new in range(new_l1):
            for y_new in range(new_l2):
                # Преобразование координат нового изображения в масштаб исходного изображения для аргументов косинуса
                x_scaled = x_new / scale_factor
                y_scaled = y_new / scale_factor

                smn_val = 0.0
                for j in range(-m, m + 1):
                    for k in range(-n, n + 1):
                        abs_j = abs(j)
                        abs_k = abs(k)

                        if abs_j <= max_calc_m and abs_k <= max_calc_n:
                            coeff = a_coefficients.get((abs_j, abs_k), 0.0)
                            # Настройка аргументов косинуса на основе предположения, что x в формуле соответствует столбцу (original_l2), а y соответствует строке (original_l1)
                            smn_val += coeff * np.cos(np.pi * j * x_scaled / original_l2) * np.cos(np.pi * k * y_scaled / original_l1)

                reconstructed_image[y_new, x_new] = smn_val

        # Ограничение значений в диапазоне 0-255 и преобразование в uint8
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        return reconstructed_image

    # --- Применение увеличения изображения (пример) ---

    # Определение параметров увеличения
    enlargement_factor = 2 # Пример: увеличение в 2 раза
    m_sum = 30 
    n_sum = 30 

    print(f"Calculating Fourier coefficients for m up to {m_sum}, n up to {n_sum}")
    # Вычисление коэффициентов на основе исходного изображения в оттенках серого и желаемых пределов суммирования m и n
    a_coeffs, orig_l1, orig_l2 = calculate_fourier_coefficients(gray_image, m_sum, n_sum)

    print(f"Reconstructing image with enlargement factor {enlargement_factor}, m={m_sum}, n={n_sum}")
    # Реконструкция изображения с использованием вычисленных коэффициентов и желаемого коэффициента увеличения
    enlarged_image_fourier = reconstruct_image_fourier(a_coeffs, orig_l1, orig_l2, m_sum, n_sum, enlargement_factor)

    # --- Сравнение со стандартной интерполяцией (например, билинейной) ---
    new_width = int(width * enlargement_factor)
    new_height = int(height * enlargement_factor)
    enlarged_image_bilinear = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # --- Отображение увеличенных изображений ---
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

    # --- Шаг 9: Спектр Фурье ---

    print("Calculating and displaying Fourier spectrum...")

    # Вычисление 2D ДПФ изображения в оттенках серого
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Сдвиг нулевой частотной составляющей в центр
    dft_shift = np.fft.fftshift(dft)

    # Вычисление спектра амплитуд (логарифмическая шкала для лучшей визуализации)
    # Амплитуда = sqrt(real**2 + imaginary**2)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Отображение спектра амплитуд
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([]), plt.yticks([]) # Скрытие значений делений
    plt.show()

    # --- Шаг 10: Фильтрация верхних и нижних частот в области Фурье ---

    print("Applying High-pass and Low-pass filters in Fourier domain...")

    # Создание масок для фильтрации
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2 # Центр спектра

    # Фильтр нижних частот (Идеальный фильтр нижних частот - простой пример)
    # Создание маски с радиусом. Частоты внутри радиуса сохраняются, снаружи удаляются.
    # Настраиваемый пользователем параметр: lowpass_radius
    lowpass_radius = 30 # Пример параметра (настройте по необходимости)
    lowpass_mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(lowpass_mask, (ccol, crow), lowpass_radius, (1, 1), -1)

    # Применение маски нижних частот к сдвинутому спектру
    fourier_lowpass = dft_shift * lowpass_mask

    # Обратное ДПФ для получения отфильтрованного изображения
    fourier_lowpass_ishift = np.fft.ifftshift(fourier_lowpass)
    image_lowpass = cv2.idft(fourier_lowpass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    image_lowpass = np.clip(image_lowpass, 0, 255).astype(np.uint8)

    # Фильтр верхних частот (Идеальный фильтр верхних частот - простой пример)
    # Создание маски. Частоты снаружи радиуса сохраняются, внутри удаляются.
    # Это инверсия маски нижних частот.
    # Настраиваемый пользователем параметр: highpass_radius (может быть таким же, как lowpass_radius, или другим)
    highpass_radius = 30 # Пример параметра (настройте по необходимости)
    highpass_mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(highpass_mask, (ccol, crow), highpass_radius, (0, 0), -1) # Рисование черного круга (0) на белом фоне (1)

    # Применение маски верхних частот к сдвинутому спектру
    fourier_highpass = dft_shift * highpass_mask

    # Обратное ДПФ для получения отфильтрованного изображения
    fourier_highpass_ishift = np.fft.ifftshift(fourier_highpass)
    image_highpass = cv2.idft(fourier_highpass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    image_highpass = np.clip(image_highpass, 0, 255).astype(np.uint8)

    # Отображение отфильтрованных изображений
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