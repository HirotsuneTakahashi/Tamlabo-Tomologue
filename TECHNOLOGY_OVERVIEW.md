# 🛠️ 技術仕様書 - 複数音声ファイル文字起こし・分析プログラム

このドキュメントでは、`multi_audio_transcriber.py` で使用されている各技術とライブラリの詳細を説明します。

## 📋 目次

1. [音声処理技術](#音声処理技術)
2. [AI・機械学習技術](#ai機械学習技術)
3. [データ処理・分析技術](#データ処理分析技術)
4. [可視化技術](#可視化技術)
5. [ファイル・システム処理技術](#ファイルシステム処理技術)
6. [ユーザーインターフェース技術](#ユーザーインターフェース技術)
7. [パフォーマンス最適化技術](#パフォーマンス最適化技術)

---

## 🎵 音声処理技術

### 1. pydub
**役割**: 音声ファイルの形式変換とメタデータ処理

```python
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
```

**機能**:
- **ファイル形式変換**: m4a → wav形式への変換
- **音声長取得**: ファイルの総時間を測定
- **無音検出**: 実際の発話部分と無音部分を区別

**技術的詳細**:
- FFmpegライブラリを内部使用
- 様々な音声形式に対応（m4a, mp3, wav, etc.）
- デシベル単位での音量解析

**使用例**:
```python
# m4a → wav変換
audio = AudioSegment.from_file("input.m4a")
audio.export("output.wav", format="wav")

# 無音部分の検出
nonsilent_ranges = detect_nonsilent(
    audio,
    min_silence_len=500,      # 0.5秒以上の無音を検出
    silence_thresh=audio.dBFS-16  # 平均音量-16dBを無音判定
)
```

### 2. torch & torchaudio
**役割**: 低レベル音声データ処理とテンソル操作

```python
import torch
import torchaudio
```

**機能**:
- **音声テンソル変換**: 音声データをPyTorchテンソルに変換
- **サンプリングレート調整**: 音声の品質調整
- **バッチ処理対応**: 複数ファイルの効率的処理

**技術的詳細**:
- CPUでの高速テンソル演算
- GPU非使用設定での安定動作
- メモリ効率的な音声データハンドリング

---

## 🤖 AI・機械学習技術

### 1. OpenAI Whisper
**役割**: 最先端の音声認識AI

```python
import whisper
```

**技術仕様**:
- **モデルサイズ**: Medium（精度と速度のバランス）
- **対応言語**: 日本語特化設定
- **処理方式**: Transformer ベースのエンドツーエンド学習

**アーキテクチャ**:
```
音声入力 → メルスペクトログラム → Encoder → Decoder → テキスト出力
         ↑                    ↑        ↑
    前処理変換              注意機構   言語モデル
```

**最適化設定**:
```python
result = self.whisper_model.transcribe(
    str(audio_path),
    language="ja",        # 日本語指定で精度向上
    task="transcribe",    # 翻訳ではなく文字起こし
    verbose=False         # 詳細ログ無効化
)
```

**精度向上のポイント**:
- 事前に音声をWAV形式に統一
- 日本語言語設定で認識精度を最大化
- Mediumモデルで処理速度と精度のバランス

---

## 📊 データ処理・分析技術

### 1. NumPy
**役割**: 高速数値計算エンジン

```python
import numpy as np
```

**使用場面**:
- **統計計算**: 平均、最大値、最小値の算出
- **配列操作**: 大量データの効率的処理
- **数学関数**: 発話速度・割合の計算

**計算例**:
```python
# 発話速度計算 (文字数/10秒)
speech_rate = (char_count / speech_time) * 10

# 統計情報
average_rate = np.mean(speech_rates)
max_rate = max(speech_rates)
```

### 2. Pandas
**役割**: データ分析・CSV処理のスイスアーミーナイフ

```python
import pandas as pd
```

**機能**:
- **データフレーム作成**: 構造化データの管理
- **CSV出力**: UTF-8エンコーディング対応
- **データ集計**: 複数ファイルの結果統合

**データ構造例**:
```python
# 発話速度データフレーム
df = pd.DataFrame({
    "人名": ["岸本智行", "田中太郎"],
    "発話速度": [45.2, 38.7],
    "文字数": [1250, 980]
})
```

---

## 📈 可視化技術

### 1. Matplotlib
**役割**: 高品質グラフ作成エンジン

```python
import matplotlib.pyplot as plt
```

**グラフ作成技術**:
- **棒グラフ**: 発話速度・割合の比較
- **カスタマイズ**: 色、フォント、レイアウト調整
- **高解像度出力**: 300dpi PNG形式

### 2. Seaborn
**役割**: 統計的可視化の美化

```python
import seaborn as sns
```

**スタイル設定**:
```python
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")  # 色相環ベースの美しいカラーパレット
```

### 3. japanize-matplotlib
**役割**: 日本語フォント対応

```python
import japanize_matplotlib
```

**OS別フォント自動選択**:
```python
if system_name == "Darwin":  # macOS
    matplotlib.rcParams['font.family'] = ['Hiragino Sans']
elif system_name == "Windows":  # Windows
    matplotlib.rcParams['font.family'] = ['Meiryo']
else:  # Linux
    matplotlib.rcParams['font.family'] = ['IPAexGothic']
```

---

## 📁 ファイル・システム処理技術

### 1. Pathlib
**役割**: モダンなファイルパス操作

```python
from pathlib import Path
```

**利点**:
- **OS非依存**: Windows, macOS, Linux対応
- **直感的API**: `/` 演算子でパス結合
- **メソッドチェーン**: `path.parent.name` など

**使用例**:
```python
INPUT_DIR = Path("input")
audio_files = list(INPUT_DIR.glob("*.m4a"))  # パターンマッチング
output_path = OUTPUT_DIR / f"{name}_結果.csv"  # パス結合
```

### 2. tempfile
**役割**: 一時ファイル管理

```python
import tempfile
```

**安全な一時ファイル処理**:
```python
TEMP_DIR = Path(tempfile.gettempdir()) / "multi_audio_transcriber"
temp_wav = TEMP_DIR / f"{audio_file.stem}.wav"

# 自動クリーンアップ
try:
    temp_wav.unlink()  # ファイル削除
except FileNotFoundError:
    pass  # 既に削除済みの場合は無視
```

### 3. 正規表現 (re)
**役割**: ファイル名パターン抽出

```python
import re
```

**人名抽出ロジック**:
```python
# "audio岸本智行21315332608.m4a" → "岸本智行"
match = re.search(r'audio([^0-9]+)', filename)
if match:
    return match.group(1).strip()
```

---

## 🖥️ ユーザーインターフェース技術

### 1. tqdm
**役割**: 美しいプログレスバー表示

```python
from tqdm.auto import tqdm
```

**機能**:
- **進捗可視化**: リアルタイム処理状況表示
- **推定時間**: 残り時間の予測
- **スループット**: 処理速度の表示

**使用例**:
```python
for audio_file in tqdm(audio_files, desc="ファイル処理進行中"):
    # 処理内容
    pass
```

**出力例**:
```
ファイル処理進行中: 100%|████████████| 3/3 [15:30<00:00, 310.2s/it]
```

### 2. コンソール出力デザイン
**役割**: ユーザーフレンドリーな情報表示

**デザインパターン**:
```python
print("🎙️ 複数音声ファイル文字起こし・分析プログラム開始")
print("=" * 60)  # 区切り線
print("📂 見つかったファイル: 3個")
print("   ✅ 処理完了")  # 成功表示
print("   ❌ エラー:")   # エラー表示
```

---

## ⚡ パフォーマンス最適化技術

### 1. CPU最適化
**設定項目**:
```python
# スレッド数制限（安定性重視）
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
```

**理由**:
- メモリ競合の回避
- CPU使用率の安定化
- 予期しないクラッシュの防止

### 2. メモリ管理
**戦略**:
- **遅延読み込み**: Whisperモデルを使用時に読み込み
- **一時ファイル管理**: 処理完了後の自動削除
- **ガベージコレクション**: 大きなオブジェクトの適切な解放

### 3. エラーハンドリング
**多層防御**:
```python
try:
    # メイン処理
    result = process_audio()
except SpecificException as e:
    # 特定エラーの対処
    handle_specific_error(e)
except Exception as e:
    # 汎用エラー処理
    handle_general_error(e)
finally:
    # クリーンアップ処理
    cleanup_resources()
```

---

## 🔐 セキュリティ・プライバシー技術

### 1. SSL証明書管理
```python
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()
```

### 2. ローカル処理
- **音声データ**: 外部送信なし、ローカル処理のみ
- **キャッシュ**: ローカルファイルシステムに保存
- **結果データ**: すべてローカルに出力

---

## 📦 依存関係マネジメント

### requirements.txt 構成
```
# 音声処理
openai-whisper>=20231117
pydub>=0.25.1
torch>=2.0.0
torchaudio>=2.0.0

# データ処理・分析  
numpy>=1.24.0
pandas>=2.0.0

# 可視化
matplotlib>=3.7.0
seaborn>=0.12.0
japanize-matplotlib>=1.1.3

# ユーザーインターフェース
tqdm>=4.65.0

# システム・セキュリティ
certifi>=2023.7.22
pathlib>=1.0.1
```

### バージョン戦略
- **下位互換性**: 最小バージョン指定で柔軟性を確保
- **安定性重視**: 実績のあるバージョンを選択
- **セキュリティ**: 最新のセキュリティアップデートに対応

---

## 🏗️ アーキテクチャ設計

### クラス設計パターン
```python
class MultiAudioTranscriber:
    """
    単一責任原則に基づく設計
    - 初期化: __init__
    - 処理実行: run
    - 個別機能: process_single_file, transcribe_audio, etc.
    """
```

### データフロー
```
入力ファイル検索 → ファイル別処理 → 結果統合 → 可視化 → 出力
     ↓              ↓              ↓        ↓      ↓
   glob()      process_single()  aggregate() plot() save()
```

### エラー処理階層
1. **ファイルレベル**: 個別ファイル処理エラー
2. **システムレベル**: リソース不足、権限エラー
3. **ユーザーレベル**: 入力形式エラー、設定ミス

---

## 🚀 将来拡張のための技術基盤

### 1. モジュール化設計
現在の設計により、以下の拡張が容易：
- 新しい音声形式対応
- 追加の分析指標
- 異なる可視化形式

### 2. 設定外部化
```python
# 将来的な設定ファイル対応
CONFIG = {
    "whisper_model": "medium",
    "silence_thresh_offset": -16,
    "min_silence_len": 500
}
```

### 3. プラグインシステム
```python
# 将来的な分析プラグイン対応
def register_analyzer(name, analyzer_func):
    ANALYZERS[name] = analyzer_func
```

---

**ドキュメント情報:**
- 作成日: 2024年
- 対象バージョン: multi_audio_transcriber.py v2.0
- 技術レベル: 中級〜上級
- 更新頻度: プログラム更新に合わせて随時更新 