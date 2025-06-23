#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数音声ファイル文字起こし・発話分析プログラム (Multi Audio Transcriber)

このプログラムは複数のm4aファイルを一括処理して、以下の分析を行います：
1. 各音声ファイルの文字起こし
2. 発話時間の測定（無音部分を除く）
3. 発話割合の計算（発話時間/総ファイル時間）
4. 発話速度の計算（文字数/10秒）

使用方法：
1. inputフォルダに複数のm4aファイルを配置
   - ファイル名形式: 「audio岸本智行21315332608.m4a」
2. このプログラムを実行： python multi_audio_transcriber.py
3. outputフォルダに分析結果が保存されます

出力：
- 各ファイルの文字起こしCSV
- 発話速度まとめ（グラフ + CSV）
- 発話割合まとめ（グラフ + CSV）

作成者: AI Assistant
更新日: 2024年
"""

# ==============================================================================
# 1. 必要なライブラリのインポート
# ==============================================================================

import os
import sys
import time
import tempfile
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import platform
import matplotlib

# 音声処理関連
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torch
import torchaudio
import whisper

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

# フォント設定
try:
    system_name = platform.system()
    if system_name == "Darwin":  # macOSの場合
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'IPAexGothic', 'sans-serif']
    elif system_name == "Windows":  # Windowsの場合
        matplotlib.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'IPAexGothic', 'sans-serif']
    else:  # Linuxやその他のOSの場合
        matplotlib.rcParams['font.family'] = ['IPAexGothic', 'Noto Sans CJK JP', 'TakaoPGothic', 'sans-serif']
    
    print(f"OS: {system_name}, 設定されたフォントファミリー: {matplotlib.rcParams['font.family']}")

except Exception as e:
    print(f"フォント設定中にエラーが発生しました: {e}")

# ==============================================================================
# 3. ファイルパスの定義
# ==============================================================================

# 入力・出力ディレクトリ
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
CACHE_DIR = Path("cache")

# 一時ファイル用ディレクトリ
TEMP_DIR = Path(tempfile.gettempdir()) / "multi_audio_transcriber"
TEMP_DIR.mkdir(exist_ok=True)

# ==============================================================================
# 4. メインクラスの定義
# ==============================================================================

class MultiAudioTranscriber:
    """
    複数音声ファイル文字起こし・分析クラス
    
    このクラスは以下の機能を提供します：
    1. 複数m4aファイルの一括処理
    2. 音声の文字起こし（Whisper使用）
    3. 発話時間の測定
    4. 発話速度・発話割合の計算
    5. 結果の可視化とファイル出力
    """
    
    def __init__(self):
        """
        クラスの初期化
        必要な設定とツールを準備します
        """
        print("=== 複数音声ファイル文字起こし・分析プログラム初期化中 ===")
        
        # Whisperモデルは使用時に遅延読み込み（メモリ効率のため）
        self.whisper_model = None
        
        # 出力ディレクトリの作成
        OUTPUT_DIR.mkdir(exist_ok=True)
        CACHE_DIR.mkdir(exist_ok=True)
        
        print("初期化完了！\n")

    def _load_whisper_model(self):
        """
        Whisper音声認識モデルを遅延読み込み
        """
        if self.whisper_model is None:
            print("   🤖 Whisper音声認識モデルを読み込み中...")
            print("   （初回実行時はダウンロードのため数分かかります）")
            try:
                # mediumモデルを使用（精度と速度のバランスが良い）
                self.whisper_model = whisper.load_model("medium")
                print("   ✅ Whisperモデル読み込み完了")
            except Exception as e:
                print(f"   ❌ エラー: Whisperモデルの読み込みに失敗しました: {e}")
                sys.exit(1)

    def extract_name_from_filename(self, filename: str) -> str:
        """
        ファイル名から人名を抽出
        
        Args:
            filename (str): ファイル名（例: "audio岸本智行21315332608.m4a"）
            
        Returns:
            str: 抽出された人名（例: "岸本智行"）
        """
        # "audio"の後から数字が始まる直前までを抽出
        match = re.search(r'audio([^0-9]+)', filename)
        if match:
            return match.group(1).strip()
        else:
            # パターンが一致しない場合は拡張子を除いたファイル名を返す
            return Path(filename).stem

    def find_audio_files(self) -> List[Path]:
        """
        inputディレクトリからm4aファイルを検索
        
        Returns:
            List[Path]: 見つかったm4aファイルのパスリスト
        """
        if not INPUT_DIR.exists():
            print(f"❌ エラー: 入力ディレクトリが見つかりません → {INPUT_DIR}")
            sys.exit(1)
        
        audio_files = list(INPUT_DIR.glob("*.m4a"))
        
        if not audio_files:
            print(f"❌ エラー: m4aファイルが見つかりません")
            print(f"   {INPUT_DIR} ディレクトリにm4aファイルを配置してください")
            sys.exit(1)
        
        if len(audio_files) > 10:
            print(f"⚠️ 警告: 10個を超えるファイルが見つかりました（{len(audio_files)}個）")
            print("   最初の10個のファイルのみ処理します")
            audio_files = audio_files[:10]
        
        return sorted(audio_files)

    def convert_audio_to_wav(self, src: Path, dst: Path) -> float:
        """
        音声ファイルをWAV形式に変換
        
        Args:
            src (Path): 変換元ファイルパス（m4a形式）
            dst (Path): 変換先ファイルパス（wav形式）
            
        Returns:
            float: 音声ファイルの長さ（秒）
        """
        try:
            # pydubを使用してm4a→wav変換
            audio = AudioSegment.from_file(src)
            audio.export(dst, format="wav")
            
            return len(audio) / 1000.0  # ミリ秒を秒に変換
            
        except Exception as e:
            print(f"   ❌ エラー: 音声ファイルの変換に失敗しました: {e}")
            raise

    def measure_speech_time(self, audio_path: Path) -> float:
        """
        音声ファイルから実際の発話時間を測定（無音部分を除く）
        
        Args:
            audio_path (Path): 音声ファイルのパス
            
        Returns:
            float: 発話時間（秒）
        """
        try:
            # 音声を読み込み
            audio = AudioSegment.from_file(audio_path)
            
            # 無音でない部分を検出
            # min_silence_len: 最小無音時間（ミリ秒）
            # silence_thresh: 無音判定の閾値（dB）
            nonsilent_ranges = detect_nonsilent(
                audio, 
                min_silence_len=500,  # 0.5秒以上の無音を判定
                silence_thresh=audio.dBFS-16  # 平均音量から16dB下を無音判定
            )
            
            # 発話時間の合計を計算
            total_speech_time = 0
            for start_ms, end_ms in nonsilent_ranges:
                total_speech_time += (end_ms - start_ms)
            
            return total_speech_time / 1000.0  # ミリ秒を秒に変換
            
        except Exception as e:
            print(f"   ⚠️ 警告: 発話時間測定に失敗しました: {e}")
            return 0.0

    def transcribe_audio_with_whisper(self, audio_path: Path) -> str:
        """
        Whisperを使用して音声を文字起こし
        
        Args:
            audio_path (Path): 音声ファイルのパス（wav形式）
            
        Returns:
            str: 文字起こし結果のテキスト
        """
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
            
            # 全セグメントのテキストを結合
            full_text = result["text"].strip()
            
            return full_text
            
        except Exception as e:
            print(f"   ❌ エラー: 音声認識に失敗しました: {e}")
            return ""

    def calculate_speech_rate(self, text: str, speech_time: float) -> float:
        """
        発話速度を計算（文字数/10秒）
        
        Args:
            text (str): 文字起こし結果
            speech_time (float): 発話時間（秒）
            
        Returns:
            float: 発話速度（文字数/10秒）
        """
        if speech_time <= 0:
            return 0.0
        
        char_count = len(text)
        # 10秒あたりの文字数を計算
        speech_rate = (char_count / speech_time) * 10
        
        return speech_rate

    def process_single_file(self, audio_file: Path) -> Dict:
        """
        単一の音声ファイルを処理
        
        Args:
            audio_file (Path): 処理対象の音声ファイル
            
        Returns:
            Dict: 処理結果の辞書
        """
        name = self.extract_name_from_filename(audio_file.name)
        print(f"\n📁 処理中: {audio_file.name} (名前: {name})")
        
        # 一時WAVファイルのパス
        temp_wav = TEMP_DIR / f"{audio_file.stem}.wav"
        
        try:
            # 1. 音声ファイル変換
            print("   🔄 音声ファイル変換中...")
            total_duration = self.convert_audio_to_wav(audio_file, temp_wav)
            
            # 2. 発話時間測定
            print("   ⏰ 発話時間測定中...")
            speech_time = self.measure_speech_time(audio_file)
            
            # 3. 文字起こし
            print("   🎙️ 文字起こし中...")
            transcribed_text = self.transcribe_audio_with_whisper(temp_wav)
            
            # 4. 発話速度計算
            speech_rate = self.calculate_speech_rate(transcribed_text, speech_time)
            
            # 5. 発話割合計算
            speech_ratio = (speech_time / total_duration) * 100 if total_duration > 0 else 0
            
            # 結果辞書を作成
            result = {
                "name": name,
                "filename": audio_file.name,
                "total_duration": total_duration,
                "speech_time": speech_time,
                "speech_ratio": speech_ratio,
                "transcribed_text": transcribed_text,
                "character_count": len(transcribed_text),
                "speech_rate": speech_rate
            }
            
            print(f"   ✅ 処理完了 - 発話時間: {speech_time:.1f}秒, 発話速度: {speech_rate:.1f}文字/10秒")
            
            return result
            
        except Exception as e:
            print(f"   ❌ エラー: ファイル処理に失敗しました: {e}")
            return {
                "name": name,
                "filename": audio_file.name,
                "total_duration": 0,
                "speech_time": 0,
                "speech_ratio": 0,
                "transcribed_text": "",
                "character_count": 0,
                "speech_rate": 0
            }
        
        finally:
            # 一時ファイルの削除
            try:
                temp_wav.unlink()
            except FileNotFoundError:
                pass

    def save_transcription_csv(self, result: Dict):
        """
        文字起こし結果をCSVファイルに保存
        
        Args:
            result (Dict): 処理結果の辞書
        """
        filename = f"{result['name']}_文字起こし.csv"
        filepath = OUTPUT_DIR / filename
        
        # 単純なテキストとして保存
        df = pd.DataFrame([{
            "ファイル名": result["filename"],
            "人名": result["name"],
            "文字起こし結果": result["transcribed_text"]
        }])
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"   💾 文字起こし保存: {filename}")

    def create_speech_rate_chart(self, results: List[Dict]):
        """
        発話速度の比較チャートを作成
        
        Args:
            results (List[Dict]): 全ファイルの処理結果
        """
        if not results:
            return
        
        # データの準備
        names = [r["name"] for r in results]
        speech_rates = [r["speech_rate"] for r in results]
        
        # グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 棒グラフ
        bars = ax.bar(names, speech_rates, color='skyblue', alpha=0.8, edgecolor='navy')
        
        # 各棒の上に数値を表示
        for bar, rate in zip(bars, speech_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rate:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # カスタマイズ
        ax.set_title('発話速度比較（文字数/10秒）', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('人名', fontsize=12)
        ax.set_ylabel('発話速度（文字数/10秒）', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # X軸のラベルを斜めに表示（長い名前の場合）
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '発話速度比較.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📊 発話速度グラフ保存: 発話速度比較.png")

    def create_speech_ratio_chart(self, results: List[Dict]):
        """
        発話割合の比較チャートを作成
        
        Args:
            results (List[Dict]): 全ファイルの処理結果
        """
        if not results:
            return
        
        # データの準備
        names = [r["name"] for r in results]
        speech_ratios = [r["speech_ratio"] for r in results]
        
        # グラフ作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 棒グラフ
        bars = ax.bar(names, speech_ratios, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        
        # 各棒の上に数値を表示
        for bar, ratio in zip(bars, speech_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # カスタマイズ
        ax.set_title('発話割合比較（発話時間/総時間）', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('人名', fontsize=12)
        ax.set_ylabel('発話割合（%）', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # X軸のラベルを斜めに表示（長い名前の場合）
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '発話割合比較.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📊 発話割合グラフ保存: 発話割合比較.png")

    def save_summary_csvs(self, results: List[Dict]):
        """
        サマリーCSVファイルを保存
        
        Args:
            results (List[Dict]): 全ファイルの処理結果
        """
        if not results:
            return
        
        # 発話速度サマリー
        speech_rate_data = []
        for r in results:
            speech_rate_data.append({
                "人名": r["name"],
                "ファイル名": r["filename"],
                "発話速度（文字数/10秒）": f"{r['speech_rate']:.1f}",
                "文字数": r["character_count"],
                "発話時間（秒）": f"{r['speech_time']:.1f}"
            })
        
        df_speech_rate = pd.DataFrame(speech_rate_data)
        df_speech_rate.to_csv(OUTPUT_DIR / '発話速度まとめ.csv', index=False, encoding='utf-8-sig')
        print("   💾 発話速度まとめ保存: 発話速度まとめ.csv")
        
        # 発話割合サマリー
        speech_ratio_data = []
        for r in results:
            speech_ratio_data.append({
                "人名": r["name"],
                "ファイル名": r["filename"],
                "発話割合（%）": f"{r['speech_ratio']:.1f}",
                "発話時間（秒）": f"{r['speech_time']:.1f}",
                "総時間（秒）": f"{r['total_duration']:.1f}"
            })
        
        df_speech_ratio = pd.DataFrame(speech_ratio_data)
        df_speech_ratio.to_csv(OUTPUT_DIR / '発話割合まとめ.csv', index=False, encoding='utf-8-sig')
        print("   💾 発話割合まとめ保存: 発話割合まとめ.csv")

    def display_summary(self, results: List[Dict]):
        """
        処理結果のサマリーを表示
        
        Args:
            results (List[Dict]): 全ファイルの処理結果
        """
        print("\n" + "=" * 60)
        print("📊 処理結果サマリー")
        print("=" * 60)
        
        if not results:
            print("処理されたファイルがありません。")
            return
        
        print(f"処理ファイル数: {len(results)}個")
        print("\n📈 各人の分析結果:")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']} ({result['filename']})")
            print(f"   📄 文字数: {result['character_count']:,}文字")
            print(f"   ⏰ 発話時間: {result['speech_time']:.1f}秒 / {result['total_duration']:.1f}秒")
            print(f"   📊 発話割合: {result['speech_ratio']:.1f}%")
            print(f"   🗣️ 発話速度: {result['speech_rate']:.1f}文字/10秒")
            print()
        
        # 統計情報
        speech_rates = [r['speech_rate'] for r in results if r['speech_rate'] > 0]
        speech_ratios = [r['speech_ratio'] for r in results if r['speech_ratio'] > 0]
        
        if speech_rates:
            print("📊 統計情報:")
            print(f"   平均発話速度: {np.mean(speech_rates):.1f}文字/10秒")
            print(f"   最高発話速度: {max(speech_rates):.1f}文字/10秒 ({results[speech_rates.index(max(speech_rates))]['name']})")
            print(f"   最低発話速度: {min(speech_rates):.1f}文字/10秒 ({results[speech_rates.index(min(speech_rates))]['name']})")
        
        if speech_ratios:
            print(f"   平均発話割合: {np.mean(speech_ratios):.1f}%")
            print(f"   最高発話割合: {max(speech_ratios):.1f}% ({results[speech_ratios.index(max(speech_ratios))]['name']})")
            print(f"   最低発話割合: {min(speech_ratios):.1f}% ({results[speech_ratios.index(min(speech_ratios))]['name']})")
        
        print("\n" + "=" * 60)
        print("✅ 全処理完了！結果は以下で確認できます：")
        print(f"📁 出力ディレクトリ: {OUTPUT_DIR}/")
        print("   - 各人の文字起こしCSV")
        print("   - 発話速度比較グラフ (PNG)")
        print("   - 発話割合比較グラフ (PNG)")
        print("   - まとめCSVファイル")
        print("=" * 60)

    def run(self):
        """
        メインの処理実行
        """
        print("🎙️ 複数音声ファイル文字起こし・分析プログラム開始")
        print("=" * 60)
        
        # 音声ファイルの検索
        audio_files = self.find_audio_files()
        print(f"📂 見つかったファイル: {len(audio_files)}個")
        for i, file in enumerate(audio_files, 1):
            name = self.extract_name_from_filename(file.name)
            print(f"   {i}. {file.name} → {name}")
        
        # 各ファイルを処理
        results = []
        print(f"\n🔄 処理開始...")
        
        for audio_file in tqdm(audio_files, desc="ファイル処理進行中"):
            result = self.process_single_file(audio_file)
            results.append(result)
            
            # 個別ファイルの文字起こしCSVを保存
            self.save_transcription_csv(result)
        
        # グラフとサマリーの作成
        print(f"\n📊 結果の可視化・保存中...")
        self.create_speech_rate_chart(results)
        self.create_speech_ratio_chart(results)
        self.save_summary_csvs(results)
        
        # 結果表示
        self.display_summary(results)
        
        # 一時ディレクトリのクリーンアップ
        try:
            for temp_file in TEMP_DIR.iterdir():
                temp_file.unlink()
            TEMP_DIR.rmdir()
        except Exception:
            pass  # 削除に失敗しても継続


# ==============================================================================
# 5. メイン実行部分
# ==============================================================================

def main():
    """
    メイン実行関数
    """
    try:
        transcriber = MultiAudioTranscriber()
        transcriber.run()
    except KeyboardInterrupt:
        print("\n\n⏹️ プログラムが中断されました")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 予期しないエラーが発生しました: {e}")
        print("エラーの詳細:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()