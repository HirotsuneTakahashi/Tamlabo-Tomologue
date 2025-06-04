#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面接音声分析プログラム (Interview Speech Analyzer)

このプログラムは面接音声ファイル（m4a形式）を分析して、以下の評価指標を算出します：
1. ポジティブ vs ネガティブワードの比率
2. 躊躇率（フィラー頻度）
3. 粘り強さの表現頻度
4. 質問応答速度
5. 接続詞使用頻度
6. 発話速度（WPM：Words Per Minute）
7. 一人称/三人称比率

使用方法：
1. inputフォルダに分析したいm4aファイルを「interview.m4a」という名前で配置
2. このプログラムを実行： python interview_analyzer.py
3. outputフォルダに分析結果（グラフ画像とCSVファイル）が保存されます

注意：
- 初回実行時はWhisperモデルやSpeechBrainモデルのダウンロードに時間がかかります
- インターネット接続が必要です
- Python 3.8以上が必要です

作成者: [あなたの名前]
更新日: 2024年
"""

# ==============================================================================
# 1. 必要なライブラリのインポート
# ==============================================================================

import os
import sys
import time
import tempfile
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import platform  # OSを判定するために追加
import matplotlib  # rcParamsを直接設定するために追加

# 音声処理関連
from pydub import AudioSegment
import torch
import torchaudio
import whisper

# 自然言語処理関連
from janome.tokenizer import Tokenizer

# 話者認識関連
from speechbrain.pretrained import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering

# 可視化関連
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# プログレスバー
from tqdm.auto import tqdm

# SSL証明書関連
import ssl
import certifi

# ==============================================================================
# 2. 初期設定
# ==============================================================================

# SSL証明書の設定（whisperモデルダウンロード用）
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()

# CPUスレッド数の制限（安定性向上のため）
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# グラフのスタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- ▼ここからフォント設定を追加 ▼ ---
try:
    system_name = platform.system()
    if system_name == "Darwin":  # macOSの場合
        # ヒラギノフォントを指定。システムにインストールされている正確な名前を確認してください。
        # 一般的なヒラギノフォントの候補を優先順にリストします。
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'IPAexGothic', 'sans-serif']
    elif system_name == "Windows":  # Windowsの場合
        # MeiryoやYu Gothicなどを指定
        matplotlib.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'IPAexGothic', 'sans-serif']
    else:  # Linuxやその他のOSの場合
        # japanize-matplotlibがIPAexGothicを適切に設定することを期待しつつ、
        # 他のフォントも候補に入れることができます。
        matplotlib.rcParams['font.family'] = ['IPAexGothic', 'Noto Sans CJK JP', 'TakaoPGothic', 'sans-serif']
    
    print(f"OS: {system_name}, 設定されたフォントファミリー: {matplotlib.rcParams['font.family']}")

except Exception as e:
    print(f"フォント設定中にエラーが発生しました: {e}")
    # フォント設定に失敗した場合でも、japanize-matplotlibのデフォルト設定に期待します。
    pass
# --- ▲ここまでフォント設定を追加 ▲ ---

# ==============================================================================
# 3. ファイルパスの定義
# ==============================================================================

# 入力ファイルのパス
AUDIO_M4A = Path("input/interview.m4a")  # 分析対象の音声ファイル
PATTERNS_YAML = Path("database/speech_patterns.yaml")  # 言語パターン定義ファイル

# 出力ファイルのパス
OUTPUT_DIR = Path("output")
CACHE_DIR = Path("cache")

# 一時ファイルのパス（音声変換用）
WAV_TMP = Path(tempfile.gettempdir()) / "interview_temp.wav"

# ==============================================================================
# 4. メインクラスの定義
# ==============================================================================

class InterviewSpeechAnalyzer:
    """
    面接音声分析クラス
    
    このクラスは以下の機能を提供します：
    1. 音声ファイルの読み込みと前処理
    2. 音声の文字起こし（Whisper使用）
    3. 話者の識別（面接官と面接者の区別）
    4. 言語パターンの分析
    5. 評価指標の算出
    6. 結果の可視化とファイル出力
    """
    
    def __init__(self):
        """
        クラスの初期化
        必要な設定とツールを準備します
        """
        print("=== 面接音声分析プログラム初期化中 ===")
        
        # 日本語形態素解析器の初期化
        print("日本語解析ツールを準備中...")
        self.tokenizer = Tokenizer()
        
        # 言語パターン（躊躇表現、ポジティブワードなど）の読み込み
        print("言語パターンデータを読み込み中...")
        self.patterns = self._load_patterns()
        
        # Whisperモデルは使用時に遅延読み込み（メモリ効率のため）
        self.whisper_model = None
        
        # 出力ディレクトリの作成
        OUTPUT_DIR.mkdir(exist_ok=True)
        CACHE_DIR.mkdir(exist_ok=True)
        
        print("初期化完了！\n")

    def _load_patterns(self) -> dict:
        """
        言語パターン定義ファイルを読み込む
        
        Returns:
            dict: 躊躇表現、ポジティブワード、ネガティブワードなどのパターン辞書
            
        Notes:
            speech_patterns.yamlファイルには以下が定義されています：
            - hesitation: 躊躇表現（えーと、あのー など）
            - persistence: 粘り強さ表現（継続した、頑張った など）
            - positive: ポジティブワード（成功、達成 など）
            - negative: ネガティブワード（失敗、困難 など）
            - conjunctions: 接続詞（しかし、そして など）
            - person: 一人称・三人称表現
            - thresholds: 各評価指標の閾値
        """
        if not PATTERNS_YAML.exists():
            print(f"エラー: 言語パターンファイルが見つかりません → {PATTERNS_YAML}")
            print("database/speech_patterns.yamlファイルが必要です。")
            sys.exit(1)
        
        try:
            with open(PATTERNS_YAML, 'r', encoding='utf-8') as f:
                patterns = yaml.safe_load(f)
            print(f"   ✓ パターンファイル読み込み完了: {len(patterns.get('patterns', {}))}種類")
            return patterns
        except Exception as e:
            print(f"エラー: パターンファイルの読み込みに失敗しました: {e}")
            sys.exit(1)

    def _load_whisper_model(self):
        """
        Whisper音声認識モデルを遅延読み込み
        
        Notes:
            初回実行時はインターネットからモデルをダウンロードするため
            時間がかかります（約1-2分）。2回目以降は高速に起動します。
        """
        if self.whisper_model is None:
            print("   Whisper音声認識モデルを読み込み中...")
            print("   （初回実行時はダウンロードのため数分かかります）")
            try:
                # mediumモデルを使用（精度と速度のバランスが良い）
                self.whisper_model = whisper.load_model("medium")
                print("   ✓ Whisperモデル読み込み完了")
            except Exception as e:
                print(f"   エラー: Whisperモデルの読み込みに失敗しました: {e}")
                sys.exit(1)

    def convert_audio_to_wav(self, src: Path, dst: Path):
        """
        音声ファイルをWAV形式に変換
        
        Args:
            src (Path): 変換元ファイルパス（m4a形式）
            dst (Path): 変換先ファイルパス（wav形式）
            
        Notes:
            WhisperやSpeechBrainはWAV形式を推奨するため、
            m4aファイルを一時的にWAV形式に変換します。
        """
        print("① 音声ファイル形式変換")
        start_time = time.time()
        
        try:
            # pydubを使用してm4a→wav変換
            audio = AudioSegment.from_file(src)
            audio.export(dst, format="wav")
            
            duration = time.time() - start_time
            print(f"   ✓ 変換完了 ({duration:.1f}秒)")
            print(f"   音声ファイル長: {len(audio)/1000:.1f}秒\n")
            
        except Exception as e:
            print(f"   エラー: 音声ファイルの変換に失敗しました: {e}")
            print("   m4aファイルが正しい形式かご確認ください。")
            sys.exit(1)

    def load_cached_results(self):
        """
        キャッシュされた分析結果を読み込む
        
        Returns:
            list or None: キャッシュされたセグメントデータ、または None（キャッシュなし）
            
        Notes:
            一度分析を実行すると結果がキャッシュされ、
            次回実行時は高速に結果を表示できます。
            新しい音声ファイルで分析したい場合は、
            cacheフォルダを削除してください。
        """
        cache_file = CACHE_DIR / "analysis_cache.csv"
        
        if not cache_file.exists():
            return None
            
        print("② キャッシュされた分析結果を読み込み")
        start_time = time.time()
        
        try:
            df = pd.read_csv(cache_file)
            segments = []
            
            # CSVからセグメントデータを復元
            for _, row in df.iterrows():
                segment = {
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "is_interviewer": row["is_interviewer"],
                    "metrics": {
                        "tokens": row["tokens"],
                        "hesitation": row["hesitation"],
                        "persistence": row["persistence"],
                        "conjunctions": row["conjunctions"],
                        "first_person": row["first_person"],
                        "third_person": row["third_person"],
                        "positive": row.get("positive", 0),
                        "negative": row.get("negative", 0)
                    }
                }
                segments.append(segment)
            
            duration = time.time() - start_time
            print(f"   ✓ キャッシュ読み込み完了 ({duration:.1f}秒)")
            print(f"   読み込みセグメント数: {len(segments)}個")
            print("   ※ 新しい音声で分析したい場合は cacheフォルダを削除してください\n")
            
            return segments
            
        except Exception as e:
            print(f"   警告: キャッシュ読み込みに失敗しました: {e}")
            print("   新規に分析を実行します\n")
            return None

    def transcribe_audio_with_whisper(self, audio_path: Path) -> list:
        """
        Whisperを使用して音声を文字起こし
        
        Args:
            audio_path (Path): 音声ファイルのパス（wav形式）
            
        Returns:
            list: 文字起こし結果のセグメントリスト
            
        Notes:
            Whisperは高精度な音声認識AIです。
            日本語に対応しており、句読点も自動で挿入されます。
        """
        print("② 音声文字起こし（Whisper AI使用）")
        start_time = time.time()
        
        # Whisperモデルを読み込み
        self._load_whisper_model()
        
        try:
            # Whisperで音声認識実行
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="ja",  # 日本語指定
                task="transcribe",  # 文字起こしタスク
                verbose=False  # 詳細ログ無効化
            )
            
            # セグメント情報を整理
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],  # 開始時刻（秒）
                    "end": segment["end"],      # 終了時刻（秒）
                    "text": segment["text"].strip(),  # 文字起こし結果
                    "speaker": -1  # 話者は後で識別
                })
            
            duration = time.time() - start_time
            print(f"   ✓ 文字起こし完了 ({duration:.1f}秒)")
            print(f"   認識セグメント数: {len(segments)}個")
            print(f"   総文字数: {sum(len(s['text']) for s in segments)}文字\n")
            
            return segments
            
        except Exception as e:
            print(f"   エラー: 音声認識に失敗しました: {e}")
            sys.exit(1)

    def identify_speakers(self, wav_path: Path, segments: list) -> list:
        """
        音声から話者を識別（面接官 vs 面接者）
        
        Args:
            wav_path (Path): 音声ファイルのパス
            segments (list): 文字起こしセグメントリスト
            
        Returns:
            list: 話者情報が追加されたセグメントリスト
            
        Notes:
            SpeechBrainの話者認識AIを使用し、音声の特徴から
            2人の話者を自動識別します。発話回数が少ない方を
            面接官と判定します。
        """
        print("③ 話者識別（面接官 vs 面接者の区別）")
        start_time = time.time()
        
        try:
            # 音声ファイルを読み込み
            waveform, sample_rate = torchaudio.load(str(wav_path))
            
            # 16kHzにリサンプリング（SpeechBrainの推奨設定）
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            # モノラル化（複数チャンネルの場合）
            waveform = waveform.mean(dim=0)
            
            # SpeechBrain話者認識モデルの読み込み
            print("   SpeechBrain話者認識モデルを読み込み中...")
            speaker_recognition = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/pretrained",
                run_opts={"device": "cpu"}  # CPUを使用
            )
            
            # 各セグメントの音声特徴を抽出
            embeddings = []
            print("   各発話の音声特徴を抽出中...")
            
            for segment in tqdm(segments, desc="   音声特徴抽出"):
                # セグメントの音声部分を切り出し
                start_sample = int(segment["start"] * 16000)
                end_sample = int(segment["end"] * 16000)
                segment_audio = waveform[start_sample:end_sample].unsqueeze(0)
                
                # 音声特徴ベクトルを抽出
                if len(segment_audio[0]) > 0:  # 空のセグメントをスキップ
                    embedding = speaker_recognition.encode_batch(segment_audio)
                    embeddings.append(embedding.squeeze().cpu().numpy())
                else:
                    # 空のセグメントの場合はダミーベクトル
                    embeddings.append(np.zeros(192))  # ECAパモデルの出力次元
            
            # 話者クラスタリング（2人の話者に分類）
            print("   話者をクラスタリング中...")
            embeddings_array = np.stack(embeddings)
            clustering = AgglomerativeClustering(n_clusters=2)
            speaker_labels = clustering.fit_predict(embeddings_array)
            
            # 発話回数をカウントして面接官を判定
            speaker_counts = Counter(speaker_labels)
            # 発話回数が少ない方を面接官と判定
            interviewer_label = speaker_counts.most_common()[-1][0]
            
            # セグメントに話者情報を追加
            for segment, label in zip(segments, speaker_labels):
                segment["is_interviewer"] = (label == interviewer_label)
            
            # 統計情報を表示
            interviewer_count = sum(1 for s in segments if s["is_interviewer"])
            interviewee_count = len(segments) - interviewer_count
            
            duration = time.time() - start_time
            print(f"   ✓ 話者識別完了 ({duration:.1f}秒)")
            print(f"   面接官発話: {interviewer_count}回")
            print(f"   面接者発話: {interviewee_count}回\n")
            
            return segments
            
        except Exception as e:
            print(f"   エラー: 話者識別に失敗しました: {e}")
            print("   全ての発話を面接者として処理します")
            
            # エラー時は全て面接者として扱う
            for segment in segments:
                segment["is_interviewer"] = False
            
            return segments 

    def analyze_speech_patterns(self, text: str) -> dict:
        """
        発話テキストから言語パターンを分析
        
        Args:
            text (str): 分析対象の発話テキスト
            
        Returns:
            dict: 各パターンの出現回数
            
        Notes:
            以下のパターンを分析します：
            - 躊躇表現（えーと、あのー など）
            - 粘り強さ表現（頑張った、継続した など）
            - ポジティブワード（成功、達成 など）
            - ネガティブワード（失敗、困難 など）
            - 接続詞（しかし、そして など）
            - 一人称・三人称表現（私、彼 など）
        """
        # 日本語形態素解析で単語に分割
        tokens = [token.surface for token in self.tokenizer.tokenize(text)]
        
        # 各パターンの出現回数をカウント
        pattern_counts = {
            "tokens": len(tokens),  # 総単語数
            
            # 躊躇表現のカウント
            "hesitation": sum(
                hesitation_word in text 
                for hesitation_word in self.patterns["patterns"]["hesitation"]
            ),
            
            # 粘り強さ表現のカウント
            "persistence": sum(
                persistence_word in text 
                for persistence_word in self.patterns["patterns"]["persistence"]
            ),
            
            # 接続詞のカウント
            "conjunctions": sum(
                conjunction in text 
                for conjunction in self.patterns["patterns"]["conjunctions"]
            ),
            
            # 一人称表現のカウント
            "first_person": sum(
                first_person in text 
                for first_person in self.patterns["patterns"]["person"]["first_person"]
            ),
            
            # 三人称表現のカウント
            "third_person": sum(
                third_person in text 
                for third_person in self.patterns["patterns"]["person"]["third_person"]
            ),
            
            # ポジティブワードのカウント
            "positive": sum(
                positive_word in text 
                for positive_word in self.patterns["patterns"]["positive"]
            ),
            
            # ネガティブワードのカウント
            "negative": sum(
                negative_word in text 
                for negative_word in self.patterns["patterns"]["negative"]
            )
        }
        
        return pattern_counts

    def calculate_evaluation_metrics(self, segments: list) -> dict:
        """
        面接評価指標を計算
        
        Args:
            segments (list): 分析済みセグメントリスト
            
        Returns:
            dict: 7つの評価指標とその詳細データ
            
        Notes:
            計算される評価指標：
            1. ポジティブ vs ネガティブワードの比率
            2. 躊躇率（フィラー頻度）
            3. 粘り強さの表現頻度
            4. 平均応答速度
            5. 接続詞使用頻度
            6. 発話速度（WPM）
            7. 一人称/三人称比率
        """
        print("④ 評価指標計算")
        start_time = time.time()
        
        # 面接者の発話のみを抽出（面接官の発話は除外）
        interviewee_segments = [s for s in segments if not s["is_interviewer"]]
        
        if not interviewee_segments:
            print("   警告: 面接者の発話が検出されませんでした")
            return {}
        
        # 基本統計の集計
        total_tokens = sum(s["metrics"]["tokens"] for s in interviewee_segments)
        total_duration = sum(s["end"] - s["start"] for s in interviewee_segments)
        total_hesitation = sum(s["metrics"]["hesitation"] for s in interviewee_segments)
        total_persistence = sum(s["metrics"]["persistence"] for s in interviewee_segments)
        total_conjunctions = sum(s["metrics"]["conjunctions"] for s in interviewee_segments)
        total_first_person = sum(s["metrics"]["first_person"] for s in interviewee_segments)
        total_third_person = sum(s["metrics"]["third_person"] for s in interviewee_segments)
        total_positive = sum(s["metrics"]["positive"] for s in interviewee_segments)
        total_negative = sum(s["metrics"]["negative"] for s in interviewee_segments)
        
        # 応答速度の計算（面接官の質問終了から面接者の回答開始まで）
        response_times = []
        for i, segment in enumerate(segments[:-1]):
            if segment["is_interviewer"] and not segments[i+1]["is_interviewer"]:
                # 面接官→面接者の切り替わり時の応答時間
                response_time = segments[i+1]["start"] - segment["end"]
                if response_time >= 0:  # 負の値は除外
                    response_times.append(response_time)
        
        # 評価指標の計算
        metrics = {
            # === 7つの主要評価指標 ===
            
            # 1. ポジティブ vs ネガティブワードの比率
            "positive_negative_ratio": (
                total_positive / total_negative if total_negative > 0 else float('inf')
            ),
            
            # 2. 躊躇率（総単語数に対する躊躇表現の割合）
            "hesitation_rate": (
                total_hesitation / total_tokens if total_tokens > 0 else 0
            ),
            
            # 3. 粘り強さの表現頻度（1000単語あたり）
            "persistence_rate": (
                (total_persistence * 1000) / total_tokens if total_tokens > 0 else 0
            ),
            
            # 4. 平均応答速度（秒）
            "avg_response_time": (
                np.mean(response_times) if response_times else 0
            ),
            
            # 5. 接続詞使用頻度（1000単語あたり）
            "conjunction_rate": (
                (total_conjunctions * 1000) / total_tokens if total_tokens > 0 else 0
            ),
            
            # 6. 発話速度（WPM: Words Per Minute）
            "speech_rate_wpm": (
                (total_tokens * 60) / total_duration if total_duration > 0 else 0
            ),
            
            # 7. 一人称/三人称比率
            "first_third_ratio": (
                total_first_person / total_third_person if total_third_person > 0 else float('inf')
            ),
            
            # === 詳細データ ===
            "total_segments": len(interviewee_segments),
            "total_duration_minutes": total_duration / 60,
            "total_tokens": total_tokens,
            "total_hesitation": total_hesitation,
            "total_persistence": total_persistence,
            "total_conjunctions": total_conjunctions,
            "total_positive": total_positive,
            "total_negative": total_negative,
            "total_first_person": total_first_person,
            "total_third_person": total_third_person,
            "response_times": response_times
        }
        
        # 評価基準との比較（良好/普通/要改善の判定）
        thresholds = self.patterns["thresholds"]
        
        # 各指標の評価レベルを判定
        metrics.update({
            "positive_negative_evaluation": (
                "良好" if metrics["positive_negative_ratio"] >= thresholds["positive_negative_ratio"]["good"]
                else "普通" if metrics["positive_negative_ratio"] >= thresholds["positive_negative_ratio"]["acceptable"]
                else "要改善"
            ),
            
            "hesitation_evaluation": (
                "良好" if metrics["hesitation_rate"] <= thresholds["hesitation_rate"]["good"]
                else "普通" if metrics["hesitation_rate"] <= thresholds["hesitation_rate"]["acceptable"]
                else "要改善"
            ),
            
            "response_time_evaluation": (
                "良好" if metrics["avg_response_time"] <= thresholds["response_latency"]["good"]
                else "普通" if metrics["avg_response_time"] <= thresholds["response_latency"]["acceptable"]
                else "要改善"
            ),
            
            "speech_rate_evaluation": (
                "遅い" if metrics["speech_rate_wpm"] < thresholds["speech_rate"]["slow"]
                else "適正" if metrics["speech_rate_wpm"] < thresholds["speech_rate"]["fast"]
                else "速い"
            )
        })
        
        duration = time.time() - start_time
        print(f"   ✓ 評価指標計算完了 ({duration:.1f}秒)\n")
        
        return metrics 

    def create_visualization_charts(self, segments: list, metrics: dict):
        """
        分析結果の可視化チャートを作成
        
        Args:
            segments (list): 分析済みセグメントリスト
            metrics (dict): 計算済み評価指標
            
        Notes:
            以下のチャートを作成します：
            1. レーダーチャート（総合評価）
            2. パターン分析（詳細）
            3. 時系列チャート
            4. 応答速度分析
        """
        print("⑤ 可視化チャート作成")
        start_time = time.time()
        
        # チャート作成
        self._create_radar_chart(metrics)
        self._create_pattern_analysis_chart(metrics)
        self._create_timeline_chart(segments, metrics)
        self._create_response_time_chart(metrics)
        
        duration = time.time() - start_time
        print(f"   ✓ チャート作成完了 ({duration:.1f}秒)\n")

    def _create_radar_chart(self, metrics: dict):
        """評価指標レーダーチャートの作成"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 評価指標の準備
        categories = [
            'ポジティブ/\nネガティブ比',
            '躊躇率\n(低いほど良い)',
            '粘り強さ表現',
            '応答速度\n(速いほど良い)',
            '接続詞使用',
            '発話速度',
            '一人称/\n三人称比'
        ]
        
        # 各指標を0-10の範囲に正規化
        values = [
            min(metrics.get("positive_negative_ratio", 0), 10),  # 上限10
            max(0, 10 - metrics.get("hesitation_rate", 0) * 100),  # 躊躇率は反転
            min(metrics.get("persistence_rate", 0) / 10, 10),  # 10で割って正規化
            max(0, 10 - metrics.get("avg_response_time", 0)),  # 応答速度は反転
            min(metrics.get("conjunction_rate", 0) / 10, 10),  # 10で割って正規化
            min(metrics.get("speech_rate_wpm", 0) / 20, 10),  # 20で割って正規化
            min(metrics.get("first_third_ratio", 0), 10)  # 上限10
        ]
        
        # 角度の計算
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 閉じるために最初の値を末尾に追加
        angles += angles[:1]
        
        # プロット
        ax.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        
        # カスタマイズ
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
        ax.grid(True)
        
        plt.title('面接評価指標レーダーチャート', size=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '評価指標_レーダーチャート.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pattern_analysis_chart(self, metrics: dict):
        """発話パターン分析バーチャートの作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ポジティブ vs ネガティブワード
        pos_neg_data = [metrics.get("total_positive", 0), metrics.get("total_negative", 0)]
        colors1 = ['#4CAF50', '#F44336']
        ax1.bar(['ポジティブワード', 'ネガティブワード'], pos_neg_data, color=colors1)
        ax1.set_title('ポジティブ vs ネガティブワード使用数')
        ax1.set_ylabel('出現回数')
        for i, v in enumerate(pos_neg_data):
            ax1.text(i, v + max(pos_neg_data) * 0.01, str(v), ha='center', fontweight='bold')
        
        # 2. 躊躇・粘り強さ表現
        hesit_persist_data = [
            metrics.get("total_hesitation", 0),
            metrics.get("total_persistence", 0)
        ]
        colors2 = ['#FF9800', '#2196F3']
        ax2.bar(['躊躇表現', '粘り強さ表現'], hesit_persist_data, color=colors2)
        ax2.set_title('躊躇表現 vs 粘り強さ表現')
        ax2.set_ylabel('出現回数')
        for i, v in enumerate(hesit_persist_data):
            ax2.text(i, v + max(hesit_persist_data) * 0.01, str(v), ha='center', fontweight='bold')
        
        # 3. 一人称 vs 三人称
        person_data = [
            metrics.get("total_first_person", 0),
            metrics.get("total_third_person", 0)
        ]
        colors3 = ['#9C27B0', '#607D8B']
        ax3.bar(['一人称表現', '三人称表現'], person_data, color=colors3)
        ax3.set_title('一人称 vs 三人称表現使用数')
        ax3.set_ylabel('出現回数')
        for i, v in enumerate(person_data):
            ax3.text(i, v + max(person_data) * 0.01, str(v), ha='center', fontweight='bold')
        
        # 4. 発話統計
        speech_stats = [
            metrics.get("total_tokens", 0),
            metrics.get("total_conjunctions", 0),
            metrics.get("total_segments", 0)
        ]
        colors4 = ['#795548', '#FF5722', '#009688']
        bars = ax4.bar(['総単語数', '接続詞数', '発話回数'], speech_stats, color=colors4)
        ax4.set_title('発話統計')
        ax4.set_ylabel('数量')
        for i, v in enumerate(speech_stats):
            ax4.text(i, v + max(speech_stats) * 0.01, str(v), ha='center', fontweight='bold')
        
        plt.suptitle('発話パターン詳細分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '発話パターン分析.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_timeline_chart(self, segments: list, metrics: dict):
        """時系列発話パターンの作成"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 面接者の発話のみ抽出
        interviewee_segments = [s for s in segments if not s["is_interviewer"]]
        
        if not interviewee_segments:
            return
        
        times = [(s["start"] + s["end"]) / 2 for s in interviewee_segments]  # 発話中央時刻
        hesitations = [s["metrics"]["hesitation"] for s in interviewee_segments]
        positives = [s["metrics"]["positive"] for s in interviewee_segments]
        negatives = [s["metrics"]["negative"] for s in interviewee_segments]
        
        # 1. 躊躇表現の時系列変化
        ax1.plot(times, hesitations, 'o-', color='#FF6B6B', linewidth=2, markersize=6)
        ax1.set_title('発話時間における躊躇表現の変化')
        ax1.set_xlabel('時間 (秒)')
        ax1.set_ylabel('躊躇表現数')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(times, hesitations, alpha=0.3, color='#FF6B6B')
        
        # 2. ポジティブ・ネガティブワードの時系列変化
        ax2.plot(times, positives, 'o-', color='#4CAF50', linewidth=2, 
                markersize=6, label='ポジティブワード')
        ax2.plot(times, negatives, 'o-', color='#F44336', linewidth=2, 
                markersize=6, label='ネガティブワード')
        ax2.set_title('ポジティブ・ネガティブワードの時系列変化')
        ax2.set_xlabel('時間 (秒)')
        ax2.set_ylabel('ワード数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '時系列発話パターン.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_response_time_chart(self, metrics: dict):
        """応答速度分布チャートの作成"""
        response_times = metrics.get("response_times", [])
        
        if not response_times:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 応答速度のヒストグラム
        ax1.hist(response_times, bins=min(len(response_times), 10), 
                color='#3F51B5', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(response_times), color='red', linestyle='--', 
                   linewidth=2, label=f'平均: {np.mean(response_times):.1f}秒')
        ax1.set_title('応答速度分布')
        ax1.set_xlabel('応答時間 (秒)')
        ax1.set_ylabel('頻度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 応答速度の時系列変化
        ax2.plot(range(1, len(response_times) + 1), response_times, 
                'o-', color='#3F51B5', linewidth=2, markersize=8)
        ax2.axhline(np.mean(response_times), color='red', linestyle='--', 
                   linewidth=2, label=f'平均: {np.mean(response_times):.1f}秒')
        ax2.set_title('応答速度の時系列変化')
        ax2.set_xlabel('質問番号')
        ax2.set_ylabel('応答時間 (秒)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '応答速度分析.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_to_files(self, segments: list, metrics: dict):
        """
        分析結果をCSVファイルに保存
        
        Args:
            segments (list): 分析済みセグメントリスト
            metrics (dict): 計算済み評価指標
            
        Notes:
            以下のファイルを保存します：
            - cache/analysis_cache.csv: 詳細セグメントデータ（キャッシュ用）
            - output/面接分析結果_概要.csv: 評価指標サマリー
            - output/面接分析結果_詳細.csv: 全発話データ
        """
        print("⑥ 分析結果保存")
        start_time = time.time()
        
        # === キャッシュファイルの保存 ===
        CACHE_DIR.mkdir(exist_ok=True)
        cache_data = []
        
        for segment in segments:
            row = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "is_interviewer": segment["is_interviewer"],
                "tokens": segment["metrics"]["tokens"],
                "hesitation": segment["metrics"]["hesitation"],
                "persistence": segment["metrics"]["persistence"],
                "conjunctions": segment["metrics"]["conjunctions"],
                "first_person": segment["metrics"]["first_person"],
                "third_person": segment["metrics"]["third_person"],
                "positive": segment["metrics"]["positive"],
                "negative": segment["metrics"]["negative"]
            }
            cache_data.append(row)
        
        pd.DataFrame(cache_data).to_csv(CACHE_DIR / "analysis_cache.csv", index=False)
        
        # === 評価指標サマリーの保存 ===
        OUTPUT_DIR.mkdir(exist_ok=True)
        summary_data = [{
            "評価項目": "ポジティブ/ネガティブ比率",
            "数値": f"{metrics.get('positive_negative_ratio', 0):.2f}",
            "評価": metrics.get('positive_negative_evaluation', '不明'),
            "詳細": f"ポジティブ:{metrics.get('total_positive', 0)}個, ネガティブ:{metrics.get('total_negative', 0)}個"
        }, {
            "評価項目": "躊躇率",
            "数値": f"{metrics.get('hesitation_rate', 0)*100:.1f}%",
            "評価": metrics.get('hesitation_evaluation', '不明'),
            "詳細": f"躊躇表現:{metrics.get('total_hesitation', 0)}個 / 総単語数:{metrics.get('total_tokens', 0)}個"
        }, {
            "評価項目": "粘り強さ表現頻度",
            "数値": f"{metrics.get('persistence_rate', 0):.1f}/1000語",
            "評価": "良好" if metrics.get('persistence_rate', 0) > 5 else "普通" if metrics.get('persistence_rate', 0) > 2 else "要改善",
            "詳細": f"粘り強さ表現:{metrics.get('total_persistence', 0)}個"
        }, {
            "評価項目": "平均応答速度",
            "数値": f"{metrics.get('avg_response_time', 0):.1f}秒",
            "評価": metrics.get('response_time_evaluation', '不明'),
            "詳細": f"応答回数:{len(metrics.get('response_times', []))}回"
        }, {
            "評価項目": "接続詞使用頻度",
            "数値": f"{metrics.get('conjunction_rate', 0):.1f}/1000語",
            "評価": "良好" if metrics.get('conjunction_rate', 0) > 10 else "普通" if metrics.get('conjunction_rate', 0) > 5 else "要改善",
            "詳細": f"接続詞:{metrics.get('total_conjunctions', 0)}個"
        }, {
            "評価項目": "発話速度",
            "数値": f"{metrics.get('speech_rate_wpm', 0):.1f} WPM",
            "評価": metrics.get('speech_rate_evaluation', '不明'),
            "詳細": f"総発話時間:{metrics.get('total_duration_minutes', 0):.1f}分"
        }, {
            "評価項目": "一人称/三人称比率",
            "数値": f"{metrics.get('first_third_ratio', 0):.2f}",
            "評価": "良好" if 1 <= metrics.get('first_third_ratio', 0) <= 3 else "要注意",
            "詳細": f"一人称:{metrics.get('total_first_person', 0)}個, 三人称:{metrics.get('total_third_person', 0)}個"
        }]
        
        pd.DataFrame(summary_data).to_csv(OUTPUT_DIR / "面接分析結果_概要.csv", index=False, encoding='utf-8-sig')
        
        # === 詳細発話データの保存 ===
        detailed_data = []
        for i, segment in enumerate(segments):
            row = {
                "発話番号": i + 1,
                "開始時刻": f"{segment['start']:.1f}秒",
                "終了時刻": f"{segment['end']:.1f}秒",
                "発話時間": f"{segment['end'] - segment['start']:.1f}秒",
                "話者": "面接官" if segment["is_interviewer"] else "面接者",
                "発話内容": segment["text"],
                "単語数": segment["metrics"]["tokens"],
                "躊躇表現": segment["metrics"]["hesitation"],
                "粘り強さ表現": segment["metrics"]["persistence"],
                "ポジティブワード": segment["metrics"]["positive"],
                "ネガティブワード": segment["metrics"]["negative"],
                "接続詞": segment["metrics"]["conjunctions"],
                "一人称": segment["metrics"]["first_person"],
                "三人称": segment["metrics"]["third_person"]
            }
            detailed_data.append(row)
        
        pd.DataFrame(detailed_data).to_csv(OUTPUT_DIR / "面接分析結果_詳細.csv", index=False, encoding='utf-8-sig')
        
        duration = time.time() - start_time
        print(f"   ✓ ファイル保存完了 ({duration:.1f}秒)")
        print(f"   保存ファイル:")
        print(f"     - {OUTPUT_DIR}/面接分析結果_概要.csv")
        print(f"     - {OUTPUT_DIR}/面接分析結果_詳細.csv")
        print(f"     - {CACHE_DIR}/analysis_cache.csv (キャッシュ)\n")

    def display_results(self, metrics: dict, segments: list):
        """
        分析結果をコンソールに表示
        
        Args:
            metrics (dict): 計算済み評価指標
            segments (list): 分析済みセグメントリスト
        """
        print("=" * 60)
        print("📊 面接音声分析結果")
        print("=" * 60)
        
        # === 基本情報 ===
        print("\n🔍 基本情報")
        print("-" * 40)
        print(f"総発話セグメント数: {len(segments)}個")
        print(f"面接者発話セグメント数: {metrics.get('total_segments', 0)}個")
        print(f"総発話時間: {metrics.get('total_duration_minutes', 0):.1f}分")
        print(f"総単語数: {metrics.get('total_tokens', 0):,}個")
        
        # === 7つの評価指標 ===
        print("\n📈 面接評価指標")
        print("-" * 40)
        
        print(f"1. ポジティブ/ネガティブ比率: {metrics.get('positive_negative_ratio', 0):.2f} "
              f"({metrics.get('positive_negative_evaluation', '不明')})")
        print(f"   📊 ポジティブワード: {metrics.get('total_positive', 0)}個")
        print(f"   📊 ネガティブワード: {metrics.get('total_negative', 0)}個")
        
        print(f"\n2. 躊躇率: {metrics.get('hesitation_rate', 0)*100:.1f}% "
              f"({metrics.get('hesitation_evaluation', '不明')})")
        print(f"   📊 躊躇表現: {metrics.get('total_hesitation', 0)}個")
        
        print(f"\n3. 粘り強さ表現頻度: {metrics.get('persistence_rate', 0):.1f}/1000語")
        print(f"   📊 粘り強さ表現: {metrics.get('total_persistence', 0)}個")
        
        print(f"\n4. 平均応答速度: {metrics.get('avg_response_time', 0):.1f}秒 "
              f"({metrics.get('response_time_evaluation', '不明')})")
        if metrics.get('response_times'):
            print(f"   📊 応答時間範囲: {min(metrics['response_times']):.1f}秒 ～ "
                  f"{max(metrics['response_times']):.1f}秒")
            print(f"   📊 応答回数: {len(metrics['response_times'])}回")
        
        print(f"\n5. 接続詞使用頻度: {metrics.get('conjunction_rate', 0):.1f}/1000語")
        print(f"   📊 接続詞: {metrics.get('total_conjunctions', 0)}個")
        
        print(f"\n6. 発話速度: {metrics.get('speech_rate_wpm', 0):.1f} WPM "
              f"({metrics.get('speech_rate_evaluation', '不明')})")
        
        print(f"\n7. 一人称/三人称比率: {metrics.get('first_third_ratio', 0):.2f}")
        print(f"   📊 一人称: {metrics.get('total_first_person', 0)}個")
        print(f"   📊 三人称: {metrics.get('total_third_person', 0)}個")
        
        # === 発話例 ===
        print("\n💬 発話例")
        print("-" * 40)
        
        # 躊躇表現を含む発話例
        hesitation_examples = [
            s for s in segments 
            if not s["is_interviewer"] and s["metrics"]["hesitation"] > 0
        ]
        
        if hesitation_examples:
            print("🤔 躊躇表現を含む発話例:")
            for i, example in enumerate(hesitation_examples[:3]):
                text_preview = example["text"][:50] + "..." if len(example["text"]) > 50 else example["text"]
                print(f"   {i+1}. 「{text_preview}」(躊躇: {example['metrics']['hesitation']}個)")
        else:
            print("🤔 躊躇表現を含む発話は検出されませんでした。")
        
        # 粘り強さ表現を含む発話例
        persistence_examples = [
            s for s in segments 
            if not s["is_interviewer"] and s["metrics"]["persistence"] > 0
        ]
        
        if persistence_examples:
            print("\n💪 粘り強さ表現を含む発話例:")
            for i, example in enumerate(persistence_examples[:3]):
                text_preview = example["text"][:50] + "..." if len(example["text"]) > 50 else example["text"]
                print(f"   {i+1}. 「{text_preview}」(粘り強さ: {example['metrics']['persistence']}個)")
        else:
            print("\n💪 粘り強さ表現を含む発話は検出されませんでした。")
        
        print("\n" + "=" * 60)
        print("✅ 分析完了！詳細な結果は以下で確認できます：")
        print(f"📁 CSVファイル: {OUTPUT_DIR}/")
        print(f"📊 グラフ画像: {OUTPUT_DIR}/*.png")
        print("=" * 60)


# ==============================================================================
# 5. メイン実行部分
# ==============================================================================

def main():
    """
    メイン実行関数
    面接音声分析の全プロセスを実行します
    """
    print("🎙️  面接音声分析プログラム開始")
    print("=" * 60)
    
    # === 音声ファイルの存在確認 ===
    if not AUDIO_M4A.exists():
        print(f"❌ エラー: 音声ファイルが見つかりません")
        print(f"   {AUDIO_M4A} に面接音声ファイル（m4a形式）を配置してください")
        sys.exit(1)
    
    # === 分析器の初期化 ===
    analyzer = InterviewSpeechAnalyzer()
    
    # === キャッシュされた結果の確認 ===
    segments = analyzer.load_cached_results()
    
    if segments is None:
        # === 新規分析の実行 ===
        print("🔄 新規分析を実行します...")
        
        # 音声ファイルをWAV形式に変換
        analyzer.convert_audio_to_wav(AUDIO_M4A, WAV_TMP)
        
        # Whisperで音声を文字起こし
        segments = analyzer.transcribe_audio_with_whisper(WAV_TMP)
        
        # 話者識別（面接官 vs 面接者）
        segments = analyzer.identify_speakers(WAV_TMP, segments)
        
        # 各発話の言語パターンを分析
        print("④ 言語パターン分析")
        start_time = time.time()
        for segment in segments:
            segment["metrics"] = analyzer.analyze_speech_patterns(segment["text"])
        
        duration = time.time() - start_time
        print(f"   ✓ 言語パターン分析完了 ({duration:.1f}秒)\n")
        
        # 一時ファイルの削除
        try:
            WAV_TMP.unlink()
        except FileNotFoundError:
            pass
    
    else:
        # === キャッシュを使用 ===
        print("⚡ キャッシュされた結果を使用します")
    
    # === 評価指標の計算 ===
    metrics = analyzer.calculate_evaluation_metrics(segments)
    
    if not metrics:
        print("❌ エラー: 評価指標の計算に失敗しました")
        sys.exit(1)
    
    # === 可視化グラフの作成 ===
    analyzer.create_visualization_charts(segments, metrics)
    
    # === 結果の保存 ===
    analyzer.save_results_to_files(segments, metrics)
    
    # === 結果の表示 ===
    analyzer.display_results(metrics, segments)


# ==============================================================================
# 6. プログラム実行部分
# ==============================================================================

if __name__ == "__main__":
    """
    プログラムの実行エントリーポイント
    
    使用方法:
        python interview_analyzer.py
    
    必要なファイル:
        - input/interview.m4a: 分析対象の音声ファイル
        - database/speech_patterns.yaml: 言語パターン定義ファイル
    
    出力:
        - output/*.png: 可視化グラフ
        - output/*.csv: 分析結果データ
        - cache/analysis_cache.csv: 次回高速実行用キャッシュ
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  プログラムが中断されました")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 予期しないエラーが発生しました: {e}")
        print("エラーの詳細:")
        import traceback
        traceback.print_exc()
        sys.exit(1) 