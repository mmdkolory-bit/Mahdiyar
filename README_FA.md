# پروژه پیش‌بینی قیمت خانه با شبکه عصبی CNN (Conv1D)

این پروژه برای مسابقه **House Prices: Advanced Regression Techniques** در Kaggle طراحی شده است. هدف، پیش‌بینی مقدار پیوسته `SalePrice` با یک شبکه عصبی کانولوشنی یک‌بعدی است.

> **نکته علمی مهم:** این مسئله رگرسیون است. `Accuracy`، `Precision`، `Recall`، `F1-score` و ماتریس ابهام معیارهای طبقه‌بندی‌اند و مستقیماً برای پیش‌بینی قیمت پیوسته معنا ندارند. در این پروژه، معیارهای اصلی رگرسیون گزارش می‌شوند؛ معیارهای طبقه‌بندی فقط به‌صورت تحلیل جانبی و با تبدیل قیمت به سه گروه «پایین/متوسط/بالا» محاسبه می‌شوند.

## ۱. ساختار پروژه

```text
house_prices_cnn_project/
├── data/
│   ├── train.csv                 # از Kaggle دانلود شود
│   ├── test.csv                  # از Kaggle دانلود شود
│   ├── sample_submission.csv     # اختیاری
│   └── data_description.txt      # اختیاری
├── notebooks/
│   └── House_Prices_CNN_Project.ipynb
├── outputs/                      # خروجی‌ها خودکار ساخته می‌شوند
├── train.py                      # آموزش، ارزیابی و ساخت submission
├── requirements.txt
├── run_windows.bat
└── README_FA.md
```

## ۲. دریافت داده‌ها

1. وارد صفحه رسمی داده‌های مسابقه شوید:
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. وارد حساب Kaggle شوید و در صورت نیاز قوانین مسابقه را بپذیرید.
3. فایل‌ها را از بخش **Download All** دریافت و از حالت فشرده خارج کنید.
4. حداقل دو فایل `train.csv` و `test.csv` را داخل پوشه `data` قرار دهید.

فایل‌های خام داده در این بسته قرار نگرفته‌اند؛ باید از حساب Kaggle خودتان دریافت شوند.

## ۳. نصب کتابخانه‌ها

Python 3.10 یا 3.11 پیشنهاد می‌شود. در ترمینال و داخل پوشه پروژه اجرا کنید:

```bash
python -m venv .venv
```

فعال‌سازی محیط مجازی در Windows:

```bash
.venv\Scripts\activate
```

فعال‌سازی در Linux/macOS:

```bash
source .venv/bin/activate
```

نصب وابستگی‌ها:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ۴. اجرای پروژه

اجرای مدل پایه و چهار آزمایش مقایسه‌ای (در مجموع پنج پیکربندی):

```bash
python train.py --train data/train.csv --test data/test.csv --output outputs --epochs 100 --batch-size 32
```

اجرای سریع‌تر بدون آزمایش مقایسه معماری‌ها:

```bash
python train.py --train data/train.csv --test data/test.csv --output outputs --epochs 60 --batch-size 32 --skip-ablation
```

در Windows می‌توانید پس از قرار دادن فایل‌ها در `data`، فایل `run_windows.bat` را اجرا کنید.

### اجرا در Google Colab

1. پوشه پروژه را در Colab بارگذاری کنید یا ZIP آن را در محیط Colab استخراج کنید.
2. در صورت نیاز Runtime را روی GPU قرار دهید.
3. فایل‌های `train.csv` و `test.csv` را در `data/` قرار دهید.
4. در سلول‌های Notebook، نصب نیازمندی‌ها و دستور اجرای پروژه را اجرا کنید.

## ۵. روش کار مدل

### آماده‌سازی داده

- ستون هدف: `SalePrice`
- ستون `Id` به مدل داده نمی‌شود و فقط برای خروجی Kaggle حفظ می‌شود.
- ویژگی‌های عددی: جایگزینی مقدار گمشده با میانه و سپس StandardScaler.
- ویژگی‌های دسته‌ای: جایگزینی مقدار گمشده و One-Hot Encoding.
- دسته‌های ناشناخته در داده تست با `handle_unknown='ignore'` مدیریت می‌شوند.
- تقسیم داده برچسب‌دار به آموزش و اعتبارسنجی: ۸۰٪ و ۲۰٪ با seed ثابت ۴۲.
- پیش‌پردازش ابتدا فقط روی بخش آموزش fit می‌شود و سپس به اعتبارسنجی اعمال می‌شود.

### معماری شبکه CNN

ورودی پس از کدگذاری به شکل دنباله‌ای از ویژگی‌ها با یک کانال به Conv1D داده می‌شود. معماری پایه شامل:

1. سه بلوک Conv1D با ۳۲، ۶۴ و ۱۲۸ فیلتر و کرنل ۳
2. Batch Normalization
3. ReLU
4. MaxPooling1D
5. Dropout
6. Global Average Pooling
7. Dense با ۶۴ نورون
8. خروجی خطی تک‌نورونی برای `log1p(SalePrice)`

تابع هزینه MSE روی `log1p(SalePrice)` است. از Early Stopping و کاهش خودکار نرخ یادگیری نیز استفاده می‌شود.

> محدودیت علمی: داده‌های جدولی ترتیب یا همسایگی مکانی طبیعی ندارند؛ بنابراین استفاده از Conv1D در این پروژه مطابق تمرین درخواستی است، اما CNN لزوماً بهترین معماری برای داده جدولی نیست.

## ۶. معیارهای ارزیابی

### معیارهای اصلی رگرسیون

- **MSE_price**: میانگین مربع خطا در واحد قیمتِ مربع‌شده
- **RMSE_price**: ریشه MSE در واحد دلار
- **MAE_price**: میانگین قدر مطلق خطا در واحد دلار
- **R2_price**: ضریب تعیین
- **MSE_log1p** و **RMSE_log1p_Kaggle**: معیارهای خطا روی لگاریتم قیمت

معیار رسمی مسابقه Kaggle برابر RMSE بین `log(predicted SalePrice)` و `log(actual SalePrice)` است. به همین دلیل مدل با هدف لگاریتمی آموزش داده می‌شود و `RMSE_log1p_Kaggle` در گزارش درج می‌شود.

### معیارهای طبقه‌بندی تکمیلی

برای پاسخ به بخش خواسته‌شده در صورت تمرین، مرزهای یک‌سوم و دوسوم قیمت از داده آموزش محاسبه می‌شوند و هر قیمت در یکی از سه گروه پایین، متوسط یا بالا قرار می‌گیرد. سپس Accuracy، Precision ماکرو، Recall ماکرو، F1 ماکرو و ماتریس ابهام برای همین **طبقه‌بندی گروه قیمتی** محاسبه می‌شوند؛ نه برای خروجی پیوسته رگرسیون.

## ۷. بررسی اثر لایه‌ها و اجزای CNN

پنج پیکربندی با تقسیم آموزش/اعتبارسنجی و پیش‌پردازش یکسان مقایسه می‌شوند:

| پیکربندی | تغییر |
|---|---|
| CNN_3_blocks_baseline | سه بلوک، BatchNorm و Dropout |
| CNN_2_blocks | کاهش تعداد بلوک‌ها به دو |
| CNN_4_blocks | افزایش تعداد بلوک‌ها به چهار |
| without_BatchNorm | حذف Batch Normalization |
| without_Dropout | حذف Dropout |

برای مقایسه از RMSE لگاریتمی، MAE و R² روی مجموعه اعتبارسنجی استفاده می‌شود. مدل با کمترین `RMSE_log1p_Kaggle` عملکرد بهتری روی معیار اصلی مسابقه دارد.

## ۸. فایل‌های خروجی

پس از اجرای موفق، پوشه `outputs/` شامل موارد زیر است:

- `REPORT.md`: گزارش خودکار فارسی با نتایج واقعی اجرا
- `submission.csv`: فایل آماده ارسال به Kaggle با ستون‌های `Id` و `SalePrice`
- `evaluation_metrics.json`: معیارهای عددی اعتبارسنجی
- `training_history.csv`: مقادیر MSE هر epoch
- `mse_by_epoch.png`: نمودار MSE آموزش و اعتبارسنجی
- `confusion_matrix_price_bands.png` و فایل CSV متناظر
- `actual_vs_predicted.png`: نمودار قیمت واقعی در برابر پیش‌بینی
- `residual_distribution.png`: نمودار توزیع باقیمانده‌ها
- `ablation_results.csv`: جدول مقایسه معماری‌ها
- `ablation_rmse_log.png` و `ablation_r2.png`: نمودار مقایسه معماری‌ها
- `validation_predictions.csv`: پیش‌بینی‌های بخش اعتبارسنجی
- `final_cnn_model.keras`: مدل نهایی آموزش‌دیده روی تمام داده برچسب‌دار
- `preprocessor.joblib`: پیش‌پردازشگر آموزش‌دیده

## ۹. نکته کلیدی درباره test.csv

در مسابقه Kaggle، فایل `test.csv` شامل مقادیر واقعی `SalePrice` نیست. بنابراین هیچ معیار خطایی را نمی‌توان مستقیماً روی آن محاسبه کرد. برای ارزیابی علمی، بخشی از `train.csv` کنار گذاشته می‌شود و معیارها روی همان validation محاسبه می‌شوند؛ سپس مدل نهایی با تمام `train.csv` دوباره آموزش می‌بیند و برای `test.csv` فایل `submission.csv` تولید می‌کند.

## ۱۰. منبع

- صفحه رسمی مسابقه و داده‌ها: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- توضیح هدف، ستون‌ها و معیار امتیازدهی: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/description
