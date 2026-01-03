"""Stock Trading Lessons for Beginners.

Interactive lessons covering Indonesian stock market basics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json
from pathlib import Path


class LessonCategory(Enum):
    """Lesson categories."""
    BASICS = "basics"
    ANALYSIS = "analysis"
    TRADING = "trading"
    RISK = "risk"
    STOCKAI = "stockai"


@dataclass
class Lesson:
    """A single lesson."""
    id: str
    title: str
    category: LessonCategory
    order: int
    content: str
    key_points: list[str]
    examples: list[dict[str, str]] = field(default_factory=list)
    practice_command: str | None = None
    quiz_questions: list[dict] = field(default_factory=list)
    duration_minutes: int = 5


@dataclass
class LessonProgress:
    """Track user progress through lessons."""
    completed_lessons: list[str] = field(default_factory=list)
    quiz_scores: dict[str, float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def complete_lesson(self, lesson_id: str) -> None:
        if lesson_id not in self.completed_lessons:
            self.completed_lessons.append(lesson_id)
        self.last_activity = datetime.now()

    def set_quiz_score(self, lesson_id: str, score: float) -> None:
        self.quiz_scores[lesson_id] = score
        self.last_activity = datetime.now()

    def get_progress_percent(self, total_lessons: int) -> float:
        return (len(self.completed_lessons) / total_lessons) * 100

    def save(self, path: Path) -> None:
        data = {
            "completed_lessons": self.completed_lessons,
            "quiz_scores": self.quiz_scores,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LessonProgress":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            completed_lessons=data.get("completed_lessons", []),
            quiz_scores=data.get("quiz_scores", {}),
            started_at=datetime.fromisoformat(data.get("started_at", datetime.now().isoformat())),
            last_activity=datetime.fromisoformat(data.get("last_activity", datetime.now().isoformat())),
        )


# ============================================================================
# LESSON CONTENT
# ============================================================================

LESSONS = [
    # ==========================================================================
    # BASICS
    # ==========================================================================
    Lesson(
        id="basics_01_what_is_stock",
        title="Apa itu Saham?",
        category=LessonCategory.BASICS,
        order=1,
        duration_minutes=5,
        content="""
# Apa itu Saham?

**Saham** adalah bukti kepemilikan suatu perusahaan. Ketika kamu membeli saham,
kamu menjadi pemilik sebagian kecil dari perusahaan tersebut.

## Contoh Sederhana

Bayangkan ada warung bakso yang butuh modal Rp 100 juta untuk buka cabang baru.
Pemilik warung membagi kepemilikan menjadi 100.000 lembar saham @ Rp 1.000.

Jika kamu beli 1.000 lembar saham (Rp 1 juta), kamu memiliki 1% warung bakso itu!

## Di Indonesia (IDX)

Bursa Efek Indonesia (BEI/IDX) adalah tempat jual-beli saham perusahaan Indonesia.
Contoh saham terkenal:
- **BBCA** = Bank Central Asia
- **TLKM** = Telkom Indonesia
- **BBRI** = Bank Rakyat Indonesia
- **ASII** = Astra International

## Kenapa Investasi Saham?

1. **Potensi keuntungan tinggi** - Rata-rata 10-15% per tahun (jangka panjang)
2. **Mudah dimulai** - Modal minimal sekitar Rp 100.000
3. **Likuid** - Bisa dijual kapan saja saat bursa buka
4. **Passive income** - Dapat dividen dari perusahaan yang untung
""",
        key_points=[
            "Saham = bukti kepemilikan perusahaan",
            "1 lot = 100 lembar saham (aturan IDX)",
            "BEI/IDX buka Senin-Jumat, 09:00-16:00 WIB",
            "Potensi untung tinggi, tapi ada risiko rugi juga",
        ],
        examples=[
            {"symbol": "BBCA", "name": "Bank Central Asia", "sector": "Perbankan"},
            {"symbol": "TLKM", "name": "Telkom Indonesia", "sector": "Telekomunikasi"},
            {"symbol": "UNVR", "name": "Unilever Indonesia", "sector": "Consumer Goods"},
        ],
        practice_command="stockai list --index IDX30",
        quiz_questions=[
            {
                "question": "Apa yang kamu dapatkan saat membeli saham?",
                "options": ["Hutang perusahaan", "Kepemilikan sebagian perusahaan", "Jaminan keuntungan", "Gaji bulanan"],
                "correct": 1,
            },
            {
                "question": "Berapa lembar saham dalam 1 lot di IDX?",
                "options": ["10 lembar", "50 lembar", "100 lembar", "1000 lembar"],
                "correct": 2,
            },
        ],
    ),

    Lesson(
        id="basics_02_how_to_profit",
        title="Cara Dapat Untung dari Saham",
        category=LessonCategory.BASICS,
        order=2,
        duration_minutes=5,
        content="""
# Cara Dapat Untung dari Saham

Ada 2 cara utama dapat uang dari saham:

## 1. Capital Gain (Selisih Harga)

Beli murah, jual mahal!

**Contoh:**
- Beli BBRI di harga Rp 4.000 × 100 lembar = Rp 400.000
- Harga naik ke Rp 5.000
- Jual: Rp 5.000 × 100 lembar = Rp 500.000
- **Untung: Rp 100.000 (25%)**

## 2. Dividen (Bagi Hasil)

Perusahaan yang untung membagi sebagian keuntungan ke pemegang saham.

**Contoh:**
- BBRI membagi dividen Rp 300 per lembar
- Kamu punya 100 lembar
- **Dapat dividen: Rp 30.000**

## Tapi Bisa Rugi Juga!

**Capital Loss** = Beli mahal, jual murah (terpaksa karena butuh uang atau cut loss)

**Contoh Rugi:**
- Beli GOTO di Rp 400 × 1000 lembar = Rp 400.000
- Harga turun ke Rp 100
- Jika jual: Rp 100 × 1000 = Rp 100.000
- **Rugi: Rp 300.000 (75%)**

## Prinsip Dasar

> "Investasi saham itu marathon, bukan sprint"

- Jangan panik saat harga turun
- Jangan serakah saat harga naik
- Selalu punya rencana (entry, target, stop-loss)
""",
        key_points=[
            "Capital Gain = untung dari selisih harga jual-beli",
            "Dividen = bagi hasil keuntungan perusahaan",
            "Capital Loss = rugi karena harga turun",
            "Investasi jangka panjang lebih aman daripada trading harian",
        ],
        practice_command="stockai info BBRI",
        quiz_questions=[
            {
                "question": "Beli saham Rp 2.000, jual Rp 2.500. Ini disebut?",
                "options": ["Dividen", "Capital Gain", "Capital Loss", "Split"],
                "correct": 1,
            },
            {
                "question": "Dividen adalah...",
                "options": ["Bonus dari broker", "Bagi hasil keuntungan perusahaan", "Hadiah dari BEI", "Bunga bank"],
                "correct": 1,
            },
        ],
    ),

    Lesson(
        id="basics_03_lot_and_pricing",
        title="Lot, Fraksi Harga, dan Biaya",
        category=LessonCategory.BASICS,
        order=3,
        duration_minutes=5,
        content="""
# Lot, Fraksi Harga, dan Biaya Trading

## 1 Lot = 100 Lembar

Di IDX, kamu tidak bisa beli 1 lembar saham. Minimal 1 lot (100 lembar).

**Contoh Perhitungan:**
- Harga BBCA: Rp 9.500 per lembar
- Beli 1 lot = 100 × Rp 9.500 = **Rp 950.000**
- Beli 5 lot = 500 × Rp 9.500 = **Rp 4.750.000**

## Fraksi Harga (Tick Size)

Pergerakan harga saham tidak bebas, ada kelipatan minimal:

| Rentang Harga | Kelipatan |
|---------------|-----------|
| < Rp 200 | Rp 1 |
| Rp 200 - Rp 500 | Rp 2 |
| Rp 500 - Rp 2.000 | Rp 5 |
| Rp 2.000 - Rp 5.000 | Rp 10 |
| > Rp 5.000 | Rp 25 |

**Contoh:**
- BBCA di Rp 9.500 → naik/turun kelipatan Rp 25
- Bisa ke Rp 9.525, Rp 9.550, dll. Tidak bisa Rp 9.510

## Biaya Trading

1. **Fee Broker (Sekuritas):** 0.15% - 0.35% per transaksi
2. **Pajak Jual:** 0.1% (hanya saat jual)
3. **Levy BEI:** 0.043%

**Contoh Total Biaya:**
- Beli Rp 1.000.000 → fee 0.15% = Rp 1.500
- Jual Rp 1.100.000 → fee 0.15% + pajak 0.1% = Rp 2.750
- **Total biaya: Rp 4.250**

## Tips untuk Pemula

1. Pilih broker dengan fee rendah (online broker)
2. Jangan terlalu sering trading (fee menumpuk)
3. Minimal modal awal: Rp 100.000 - Rp 500.000
""",
        key_points=[
            "1 lot = 100 lembar saham",
            "Fraksi harga berbeda tergantung rentang harga",
            "Biaya trading: fee broker + pajak jual",
            "Modal awal bisa mulai dari Rp 100.000",
        ],
        practice_command="stockai list --prices",
        quiz_questions=[
            {
                "question": "Harga saham Rp 3.000. Jika naik 1 tick, harganya jadi?",
                "options": ["Rp 3.001", "Rp 3.005", "Rp 3.010", "Rp 3.025"],
                "correct": 2,
            },
            {
                "question": "Biaya yang HANYA dikenakan saat jual saham adalah?",
                "options": ["Fee broker", "Pajak 0.1%", "Levy BEI", "Semua di atas"],
                "correct": 1,
            },
        ],
    ),

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================
    Lesson(
        id="analysis_01_fundamental",
        title="Analisis Fundamental",
        category=LessonCategory.ANALYSIS,
        order=4,
        duration_minutes=7,
        content="""
# Analisis Fundamental

Analisis fundamental = menilai saham berdasarkan kesehatan & kinerja perusahaan.

## Metrik Penting

### 1. P/E Ratio (Price to Earnings)
Berapa kali lipat harga saham dibanding keuntungan per lembar.

- P/E < 10: Murah (atau ada masalah)
- P/E 10-20: Wajar
- P/E > 20: Mahal (atau sedang bertumbuh cepat)

**Rumus:** Harga Saham ÷ Laba per Lembar (EPS)

### 2. P/B Ratio (Price to Book)
Perbandingan harga saham dengan nilai buku perusahaan.

- P/B < 1: Harga di bawah nilai aset (undervalued)
- P/B 1-3: Wajar
- P/B > 3: Premium (brand kuat atau ekspektasi tinggi)

### 3. ROE (Return on Equity)
Seberapa efisien perusahaan menghasilkan laba dari modal.

- ROE > 15%: Bagus
- ROE > 20%: Sangat bagus
- ROE < 10%: Kurang efisien

### 4. Dividend Yield
Persentase dividen terhadap harga saham.

- Yield > 5%: Tinggi (bagus untuk passive income)
- Yield 2-5%: Normal
- Yield < 2%: Rendah (growth stock)

## Contoh Analisis

**BBRI (Bank BRI):**
- P/E: 12x (wajar untuk bank)
- P/B: 2.1x (premium, brand kuat)
- ROE: 18% (sangat efisien)
- Dividend Yield: 5.2% (tinggi!)

Kesimpulan: Saham bagus untuk investor yang cari dividen.
""",
        key_points=[
            "P/E Ratio = harga dibagi laba per lembar",
            "P/B < 1 bisa berarti undervalued",
            "ROE > 15% menunjukkan perusahaan efisien",
            "Dividend Yield tinggi cocok untuk passive income",
        ],
        practice_command="stockai analyze BBRI --verbose",
        quiz_questions=[
            {
                "question": "P/E Ratio 8x artinya?",
                "options": ["Sangat mahal", "Relatif murah", "Perusahaan rugi", "Tidak ada dividen"],
                "correct": 1,
            },
            {
                "question": "ROE 25% menunjukkan?",
                "options": ["Perusahaan rugi", "Efisiensi tinggi", "Saham mahal", "Utang besar"],
                "correct": 1,
            },
        ],
    ),

    Lesson(
        id="analysis_02_technical",
        title="Analisis Teknikal Dasar",
        category=LessonCategory.ANALYSIS,
        order=5,
        duration_minutes=7,
        content="""
# Analisis Teknikal Dasar

Analisis teknikal = memprediksi harga berdasarkan pola grafik dan indikator.

## Support & Resistance

### Support (Lantai)
Level harga di mana saham cenderung berhenti turun dan memantul naik.

### Resistance (Atap)
Level harga di mana saham cenderung berhenti naik dan turun kembali.

```
Harga
  ↑
  |     ___________  ← Resistance (Rp 5.000)
  |    /           \\
  |   /             \\
  |  /               \\____
  | /                     ← Support (Rp 4.000)
  |________________________→ Waktu
```

## Indikator Populer

### 1. Moving Average (MA)
Rata-rata harga selama periode tertentu.
- MA20: Rata-rata 20 hari (short-term)
- MA50: Rata-rata 50 hari (medium-term)
- MA200: Rata-rata 200 hari (long-term)

**Sinyal:**
- Harga > MA: Bullish (cenderung naik)
- Harga < MA: Bearish (cenderung turun)

### 2. RSI (Relative Strength Index)
Mengukur momentum, skala 0-100.

- RSI > 70: Overbought (sudah naik terlalu tinggi, waspada koreksi)
- RSI < 30: Oversold (sudah turun terlalu dalam, potensi rebound)
- RSI 30-70: Normal

### 3. MACD
Mengukur momentum dan arah tren.

- MACD cross up: Sinyal beli
- MACD cross down: Sinyal jual

## Pola Chart Sederhana

1. **Higher High, Higher Low** = Uptrend (tren naik)
2. **Lower High, Lower Low** = Downtrend (tren turun)
3. **Double Bottom** = Sinyal pembalikan naik
4. **Double Top** = Sinyal pembalikan turun
""",
        key_points=[
            "Support = level bawah yang menahan harga",
            "Resistance = level atas yang menahan harga",
            "RSI > 70 = overbought, RSI < 30 = oversold",
            "Moving Average membantu identifikasi tren",
        ],
        practice_command="stockai technical BBCA",
        quiz_questions=[
            {
                "question": "RSI menunjukkan angka 25. Ini berarti?",
                "options": ["Overbought", "Oversold", "Normal", "Error"],
                "correct": 1,
            },
            {
                "question": "Harga saham di atas MA200, ini menunjukkan?",
                "options": ["Bearish", "Bullish", "Sideways", "Crash"],
                "correct": 1,
            },
        ],
    ),

    # ==========================================================================
    # RISK MANAGEMENT
    # ==========================================================================
    Lesson(
        id="risk_01_basics",
        title="Manajemen Risiko",
        category=LessonCategory.RISK,
        order=6,
        duration_minutes=7,
        content="""
# Manajemen Risiko

> "Melindungi modal lebih penting daripada mencari untung"

## Aturan Emas

### 1. Jangan Taruh Semua Telur di Satu Keranjang
Diversifikasi! Beli saham dari berbagai sektor.

**Contoh Portofolio Rp 5 Juta:**
- BBRI (Bank): Rp 1.5 juta (30%)
- TLKM (Telco): Rp 1.5 juta (30%)
- ASII (Otomotif): Rp 1 juta (20%)
- UNVR (Consumer): Rp 1 juta (20%)

### 2. Stop-Loss (Batasan Rugi)
Tentukan batas kerugian SEBELUM beli saham.

**Contoh:**
- Beli BBRI di Rp 4.500
- Stop-loss: -7% = Rp 4.185
- Jika harga turun ke Rp 4.185, JUAL! Jangan berharap rebound.

### 3. Position Sizing
Jangan beli terlalu besar di satu saham.

**Aturan:**
- Maksimal 20-25% portofolio per saham
- Maksimal 2% risiko per trade

**Contoh:**
- Modal total: Rp 10 juta
- Maksimal per saham: Rp 2.5 juta (25%)
- Jika stop-loss 7%, risiko = Rp 175.000 (1.75% modal)

### 4. Risk-Reward Ratio
Potensi untung harus lebih besar dari potensi rugi.

**Target minimal 1:2**
- Risiko: Rp 100.000 (stop-loss)
- Target untung: Rp 200.000 (take profit)

## Kesalahan Umum Pemula

1. ❌ Averaging down saham jelek (menambah saat rugi)
2. ❌ Tidak pasang stop-loss
3. ❌ All-in di satu saham
4. ❌ Trading pakai uang panas (butuh segera)
5. ❌ Ikut-ikutan tanpa analisis sendiri
""",
        key_points=[
            "Diversifikasi minimal 4-5 saham berbeda sektor",
            "Selalu pasang stop-loss sebelum beli",
            "Maksimal 20-25% modal per saham",
            "Target risk-reward minimal 1:2",
        ],
        practice_command="stockai agents risk BBRI",
        quiz_questions=[
            {
                "question": "Stop-loss 7% dari harga beli Rp 5.000 adalah?",
                "options": ["Rp 4.650", "Rp 4.850", "Rp 5.350", "Rp 5.070"],
                "correct": 0,
            },
            {
                "question": "Risk-reward 1:3 artinya?",
                "options": ["Risiko 3x lebih besar", "Target untung 3x dari risiko", "Stop-loss 3%", "Beli 3 lot"],
                "correct": 1,
            },
        ],
    ),

    # ==========================================================================
    # STOCKAI USAGE
    # ==========================================================================
    Lesson(
        id="stockai_01_getting_started",
        title="Memulai dengan StockAI",
        category=LessonCategory.STOCKAI,
        order=7,
        duration_minutes=5,
        content="""
# Memulai dengan StockAI

StockAI adalah asisten AI untuk analisis saham Indonesia.

## Command Dasar

### 1. Lihat Daftar Saham
```bash
stockai list                    # IDX30
stockai list --index LQ45       # LQ45
stockai list --prices           # Dengan harga
```

### 2. Info Saham
```bash
stockai info BBCA               # Info dasar
stockai analyze BBRI            # Analisis lengkap
stockai technical TLKM          # Analisis teknikal
```

### 3. Multi-Agent Analysis
```bash
stockai agents scan             # Scan peluang
stockai agents analyze BBRI     # Analisis 7 agen
stockai agents daily 1000000    # Rekomendasi harian
stockai agents signal BBCA      # Sinyal cepat
```

### 4. Paper Trading (Latihan)
```bash
stockai paper start 10000000    # Mulai dengan Rp 10 juta virtual
stockai paper buy BBRI 2        # Beli 2 lot BBRI
stockai paper sell BBRI 1       # Jual 1 lot BBRI
stockai paper portfolio         # Lihat portofolio
```

## Workflow Harian

1. **Pagi (08:30):** `stockai agents scan` - Cari peluang
2. **Analisis:** `stockai agents analyze BBRI` - Deep dive
3. **Eksekusi:** `stockai paper buy BBRI 2` - Paper trade dulu
4. **Monitor:** `stockai paper portfolio` - Pantau posisi

## Tips

- Mulai dengan paper trading 1-2 bulan
- Jangan langsung pakai uang asli
- Catat setiap trade dan alasannya
- Review performa mingguan
""",
        key_points=[
            "Gunakan 'stockai list' untuk lihat daftar saham",
            "Gunakan 'stockai agents analyze' untuk analisis AI",
            "Mulai dengan paper trading sebelum uang asli",
            "Review dan evaluasi performa secara berkala",
        ],
        practice_command="stockai list --prices",
        quiz_questions=[
            {
                "question": "Command untuk analisis lengkap BBRI adalah?",
                "options": ["stockai info BBRI", "stockai agents analyze BBRI", "stockai list BBRI", "stockai buy BBRI"],
                "correct": 1,
            },
        ],
    ),

    Lesson(
        id="stockai_02_first_analysis",
        title="Analisis Pertama dengan StockAI",
        category=LessonCategory.STOCKAI,
        order=8,
        duration_minutes=10,
        content="""
# Analisis Pertama dengan StockAI

Mari lakukan analisis pertama kamu!

## Langkah 1: Scan Market

```bash
stockai agents scan
```

Ini akan menampilkan 5 saham dengan peluang terbaik hari ini.

## Langkah 2: Pilih Satu Saham

Dari hasil scan, pilih saham yang menarik. Contoh: BBRI

```bash
stockai agents analyze BBRI --verbose
```

Output akan menunjukkan:
- **Fundamental Score:** Kesehatan keuangan
- **Technical Score:** Kondisi chart/harga
- **Sentiment Score:** Berita dan sentimen pasar
- **Recommendation:** BUY/HOLD/SELL
- **Entry Zone:** Harga yang bagus untuk beli
- **Stop-Loss:** Batas kerugian
- **Target:** Target harga jual

## Langkah 3: Interpretasi Hasil

**Contoh Output:**
```
🎯 RECOMMENDATION: BUY
📊 Composite Score: 7.5/10

Fundamental: 8.0/10 - Strong earnings, low P/E
Technical: 7.0/10 - Above MA50, RSI normal
Sentiment: 6.5/10 - Positive news flow

Entry Zone: Rp 4.500 - Rp 4.600
Stop-Loss: Rp 4.200 (-7%)
Target 1: Rp 4.900 (+8%)
Target 2: Rp 5.200 (+14%)
```

**Cara Baca:**
- Score > 7: Layak beli
- Score 5-7: Hold/tunggu
- Score < 5: Hindari

## Langkah 4: Paper Trade

Jika yakin, coba paper trade:

```bash
stockai paper buy BBRI 2 --entry 4550 --stoploss 4200 --target 4900
```

## Latihan

Sekarang coba sendiri:
1. Jalankan `stockai agents scan`
2. Pilih 1 saham dari hasil
3. Analisis dengan `stockai agents analyze [SYMBOL]`
4. Catat rekomendasinya
""",
        key_points=[
            "Scan dulu, baru analisis detail",
            "Score > 7 layak dipertimbangkan",
            "Selalu catat entry, stop-loss, dan target",
            "Paper trade dulu sebelum uang asli",
        ],
        practice_command="stockai agents scan",
    ),
]


def get_all_lessons() -> list[Lesson]:
    """Get all available lessons."""
    return sorted(LESSONS, key=lambda x: x.order)


def get_lesson(lesson_id: str) -> Lesson | None:
    """Get a specific lesson by ID."""
    for lesson in LESSONS:
        if lesson.id == lesson_id:
            return lesson
    return None


def get_lessons_by_category(category: LessonCategory) -> list[Lesson]:
    """Get lessons filtered by category."""
    return [l for l in get_all_lessons() if l.category == category]


def get_next_lesson(current_id: str) -> Lesson | None:
    """Get the next lesson after current."""
    lessons = get_all_lessons()
    for i, lesson in enumerate(lessons):
        if lesson.id == current_id and i + 1 < len(lessons):
            return lessons[i + 1]
    return None
