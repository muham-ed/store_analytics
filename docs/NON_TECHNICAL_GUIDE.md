# دليل غير المتخصصين: كيفية إضافة سيناريو جديد إلى المنصة

أهلاً بك! تم تصميم هذا البرنامج بمعمارية الإضافات الديناميكية (Plugin Architecture)، مما يعني أنك **لن تحتاج مطلقاً إلى تعديل أي كود في البرنامج الأساسي** لإضافة سيناريو تحليلي جديد.

---

## الخطوات البسيطة لإضافة سيناريو جديد

### الخطوة 1: إنشاء ملف سيناريو جديد
في مجلد `scenarios/` داخل المشروع، أنشئ ملف Python جديد وابدأ اسمه بـ `scenario_` (مثال: `scenario_11_shelf_touch.py`).

### الخطوة 2: نسْخ القالب الأساسي للسيناريو
انسخ الكود التالي وضعه في ملفك الجديد:

```python
import numpy as np
import supervision as sv
from typing import Dict, Any
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class MyCustomScenario(BaseScenario):
    """
    وصف للسيناريو الجديد باللغة العربية
    """
    
    @property
    def scenario_name(self) -> str:
        # اسم السيناريو الذي سيظهر في قاعدة البيانات PostgreSQL
        return "my_custom_scenario"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        # هذه الدالة تعمل مرة واحدة عند بداية تشغيل المقطع
        # يمكنك هنا قراءة الإعدادات وخيارات الكاميرا
        self.fps = fps
        self.counter = 0

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        # هذه الدالة تعمل على كل إطار (Frame) في مقطع الفيديو
        # تحتوي `detections` على كل الأشخاص المكتشفين مع أرقام التتبع الخاصة بهم
        if len(detections) > 0:
            self.counter += len(detections)

    def get_final_results(self) -> Dict[str, Any]:
        # تُرجع النتائج النهائية التي سيتم حفظها تلقائياً في قاعدة البيانات
        return {
            "total_detections": self.counter
        }
```

### الخطوة 3: التشغيل والتأكيد
بمجرد حفظ الملف في مجلد `scenarios/` وإعادة تشغيل النظام، سيقوم المحرك الأساسي بتكتشاف السيناريو الجديد وتطبيقه تلقائياً وتسجيل نتائجه في جدول `scenario_results` في قاعدة البيانات PostgreSQL دون أي تدخل إضافي!
