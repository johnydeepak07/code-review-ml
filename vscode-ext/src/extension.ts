import * as vscode from 'vscode';
import axios from 'axios';

// While developing locally, point to your local FastAPI server
// After deploying to Render, change this to your live URL
const API_URL = 'https://code-review-ml.onrender.com';

interface ReviewResponse {
  readability_score: number;
  grade: string;
  cyclomatic_complexity: number;
  max_nesting_depth: number;
  naming_entropy: number;
  has_docstrings: boolean;
  num_magic_numbers: number;
  suggestions: string;
}

export function activate(context: vscode.ExtensionContext) {

  // Status bar item — shows the score for the current file in the bottom right
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.command = 'codeReview.reviewFile';
  statusBar.tooltip = 'Click to run ML code review on this file';
  context.subscriptions.push(statusBar);

  // Register the command that appears in the command palette
  const reviewCommand = vscode.commands.registerCommand(
    'codeReview.reviewFile',
    async () => {
      const editor = vscode.window.activeTextEditor;

      if (!editor) {
        vscode.window.showWarningMessage('No file is open.');
        return;
      }

      if (editor.document.languageId !== 'python') {
        vscode.window.showWarningMessage('ML Code Reviewer only supports Python files.');
        return;
      }

      const code = editor.document.getText();
      const filename = editor.document.fileName.split('/').pop() ?? 'unknown.py';

      // Show a progress spinner while waiting for the API
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'Analyzing code...',
          cancellable: false
        },
        async () => {
          try {
            const { data } = await axios.post<ReviewResponse>(
              `${API_URL}/review`,
              { code, filename },
              { timeout: 15000 }
            );

            // Update the status bar with the score
            const scorePct = Math.round(data.readability_score * 100);
            const emoji = scorePct >= 70 ? '✅' : scorePct >= 40 ? '⚠️' : '🔴';
            statusBar.text = `${emoji} ${scorePct}% (${data.grade})`;
            statusBar.show();

            // Build the results markdown content
            const content = [
              `# Code Review — ${filename}`,
              '',
              `## Score: ${scorePct}% — Grade ${data.grade}`,
              '',
              '## Structural Metrics',
              `- Cyclomatic Complexity: ${data.cyclomatic_complexity}`,
              `- Max Nesting Depth: ${data.max_nesting_depth}`,
              `- Naming Entropy: ${data.naming_entropy} (higher = more meaningful names)`,
              `- Has Docstrings: ${data.has_docstrings ? 'Yes ✅' : 'No ❌'}`,
              `- Magic Numbers: ${data.num_magic_numbers}`,
              '',
              '## Suggestions',
              data.suggestions,
              '',
              '---',
              '_ML Code Reviewer — AST analysis + XGBoost + GPT-4o-mini_'
            ].join('\n');

            // Open results in a panel next to the current file
            const doc = await vscode.workspace.openTextDocument({
              content,
              language: 'markdown'
            });

            await vscode.window.showTextDocument(doc, {
              viewColumn: vscode.ViewColumn.Beside,
              preserveFocus: true
            });

          } catch (error: any) {
            if (error.response?.status === 422) {
              vscode.window.showErrorMessage(
                `Syntax error in your Python code: ${error.response.data.detail}`
              );
            } else if (error.code === 'ECONNREFUSED') {
              vscode.window.showErrorMessage(
                'Cannot connect to Code Review API. Is the FastAPI server running on port 8000?'
              );
            } else {
              vscode.window.showErrorMessage(`Review failed: ${error.message}`);
            }
          }
        }
      );
    }
  );

  context.subscriptions.push(reviewCommand);
}

export function deactivate() {}