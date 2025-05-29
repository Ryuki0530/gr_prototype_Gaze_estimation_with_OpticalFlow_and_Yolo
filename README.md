## 開発環境

- **Python バージョン**：`3.12.6`
- **仮想環境**：`venv` の利用を推奨します。
- **依存モジュール**：必要なパッケージは [`requirements.txt`](./requirements.txt) に記載されています。

### セットアップ手順

1. 仮想環境の作成  
    ```bash
    python -m venv venv
    ```

2. 仮想環境の有効化  
    - Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. 依存モジュールのインストール  
    ```bash
    pip install -r requirements.txt
    ```