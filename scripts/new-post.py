#!/usr/bin/env python3
"""
Create a new blog post from a Notion markdown export.

Usage:
    python scripts/new-post.py <post-folder-name> <title> <markdown-file>

Example:
    python scripts/new-post.py evolution-of-agent "Evolution of Agent" ~/Downloads/export.md
"""

import sys
import os
import re
from datetime import datetime


def convert_markdown_codeblocks(content):
    """
    Convert ```markdown ... ``` blocks into <pre> HTML blocks where
    **...** is rendered as <strong>...</strong>, preserving preformatted layout
    while allowing bold text rendering.
    """
    import html as html_module

    def replace_block(match):
        inner = match.group(1)
        # Escape HTML entities so raw content is safe
        inner = html_module.escape(inner)
        # Convert **...** to <strong>...</strong>
        inner = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', inner)
        return f'<pre>{inner}</pre>'

    content = re.sub(r'```markdown\s*\n([\s\S]*?)```', replace_block, content)
    return content


def protect_math_formulas(content):
    r"""
    Replace $...$ and $$...$$ with placeholders that marked.js won't touch.
    The placeholders will be replaced with actual MathJax-compatible HTML after marked.js runs.
    """
    # First, protect code blocks from math processing
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f'\x00CODEBLOCK{len(code_blocks) - 1}\x00'

    # Save fenced code blocks (```...```)
    content = re.sub(r'```[\s\S]*?```', save_code_block, content)
    # Save inline code (`...`)
    content = re.sub(r'`[^`\n]+`', save_code_block, content)

    # Store math formulas separately
    math_formulas = []

    # Convert display math $$...$$ to placeholder
    def replace_display_math(match):
        formula = match.group(1)
        idx = len(math_formulas)
        math_formulas.append(('display', formula))
        return f'MATHPLACEHOLDER{idx}ENDMATH'

    content = re.sub(r'\$\$([\s\S]*?)\$\$', replace_display_math, content)

    # Convert inline math $...$ to placeholder
    def replace_inline_math(match):
        formula = match.group(1)
        idx = len(math_formulas)
        math_formulas.append(('inline', formula))
        return f'MATHPLACEHOLDER{idx}ENDMATH'

    content = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', replace_inline_math, content)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f'\x00CODEBLOCK{i}\x00', block)

    return content, math_formulas


TEMPLATE = '''<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <meta http-equiv="content-type" content="text/html; charset=utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} · Anna's Blog</title>

  <meta name="description" content="" />
  <meta property="og:locale" content="en"/>
  <meta property="og:image" content="https://GAOYUEtianc.github.io/img/profile1.jpg">
  <meta property="og:site_name" content="Anna's Blog"/>
  <meta property="og:title" content="{title}"/>

  <!-- MathJax 3 Configuration (must be before loading MathJax) -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
        processEscapes: true,
        processEnvironments: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <!-- MathJax 3 -->
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

  <!-- Marked.js for Markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <link type="text/css" rel="stylesheet" href="https://GAOYUEtianc.github.io/blogs/css/print.css" media="print">
  <link type="text/css" rel="stylesheet" href="https://GAOYUEtianc.github.io/blogs/css/poole.css">
  <link type="text/css" rel="stylesheet" href="https://GAOYUEtianc.github.io/blogs/css/hyde.css">

  <style type="text/css">
    .sidebar {{ background-color: #BD5D38; }}
    .read-more-link a {{ border-color: #BD5D38; }}
    footer a, .content a, .related-posts li a:hover {{ color: #BD5D38; }}
    code.has-jax {{ font: inherit; font-size: 100%; background: inherit; border: inherit; color: #515151; }}
    #markdown-content img {{ max-width: 100%; height: auto; }}
    #markdown-content pre {{ background: #f5f5f5; padding: 1em; overflow-x: auto; white-space: pre; word-break: normal; word-wrap: normal; }}
    #markdown-content code {{ background: #f5f5f5; padding: 0.2em 0.4em; }}
    #markdown-content blockquote {{ border-left: 4px solid #BD5D38; margin-left: 0; padding-left: 1em; color: #666; }}
  </style>

  <link type="text/css" rel="stylesheet" href="https://GAOYUEtianc.github.io/blogs/css/blog.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Open+Sans:400,400i,700&display=swap">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.12.1/css/all.min.css" integrity="sha256-mmgLkCYLUQbXn0B1SRqzHar6dCnv9oZFPEC1g1cwlkk=" crossorigin="anonymous" />
  <link rel="shortcut icon" href="/favicon.png">
</head>

<body>
  <aside class="sidebar">
    <div class="container">
      <div class="sidebar-about">
        <div class="author-image">
          <img src="https://GAOYUEtianc.github.io/img/profile1.jpg" class="img-circle img-headshot center" alt="Profile">
        </div>
        <h1>Anna's Blog</h1>
      </div>

      <nav>
        <ul class="sidebar-nav">
          <li><a href="https://GAOYUEtianc.github.io/blogs/">Home</a></li>
          <li><a href="https://GAOYUEtianc.github.io/">My Webpage</a></li>
        </ul>
      </nav>

      <section class="social-icons">
        <a href="https://www.linkedin.com/in/yue-anna-gao-49a834107/" rel="me" title="Linkedin">
          <i class="fab fa-linkedin" aria-hidden="true"></i>
        </a>
        <a href="https://github.com/GAOYUEtianc" rel="me" title="GitHub">
          <i class="fab fa-github" aria-hidden="true"></i>
        </a>
        <a href="https://www.facebook.com/yue.gao.925" rel="me" title="Facebook">
          <i class="fab fa-facebook" aria-hidden="true"></i>
        </a>
      </section>
    </div>
  </aside>

  <main class="content container">
    <div class="post">
      <h1>{title}</h1>
      <div class="post-date">
        <time datetime="{date_iso}">{date_display}</time>
      </div>

      <div id="markdown-content"></div>
    </div>
  </main>

  <footer>
    <div class="copyright">
      &copy; Gao Yue {year} · <a href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0</a>
    </div>
  </footer>

  <script>
    const markdownContent = `{markdown_content}`;
    const mathFormulas = {math_formulas_json};

    // First render markdown
    let html = marked.parse(markdownContent);

    // Then restore math formulas (marked.js never saw them)
    mathFormulas.forEach((item, idx) => {{
      const [type, formula] = item;
      const placeholder = `MATHPLACEHOLDER${{idx}}ENDMATH`;
      if (type === 'display') {{
        html = html.replace(placeholder, `\\\\[${{formula}}\\\\]`);
      }} else {{
        html = html.replace(placeholder, `\\\\(${{formula}}\\\\)`);
      }}
    }});

    document.getElementById('markdown-content').innerHTML = html;

    // MathJax 3 typesetting - wait for MathJax to be ready
    function typesetMath() {{
      if (window.MathJax && MathJax.typesetPromise) {{
        MathJax.typesetPromise([document.getElementById('markdown-content')]);
      }}
    }}

    // MathJax loads async, so we need to wait for it
    if (window.MathJax && MathJax.startup) {{
      MathJax.startup.promise.then(typesetMath);
    }} else {{
      // Fallback: wait for window load
      window.addEventListener('load', function() {{
        setTimeout(typesetMath, 100);
      }});
    }}
  </script>
</body>
</html>
'''

BLOG_ENTRY = '''  <article class="post">
      <h1 class="post-title">
        <a href="https://GAOYUEtianc.github.io/blogs/post/{folder}/">{title}</a>
      </h1>
      <div class="post-date">
        <time datetime="{date_iso}">{date_display}</time>
      </div>
      {description}
      <div class="read-more-link">
        <a href="/blogs/post/{folder}/">Read More</a>
      </div>
  </article>
'''

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nError: Missing arguments")
        sys.exit(1)

    folder_name = sys.argv[1]
    title = sys.argv[2]
    md_file = sys.argv[3]
    description = sys.argv[4] if len(sys.argv) > 4 else ""

    # Get script directory to find repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Read markdown file
    with open(os.path.expanduser(md_file), 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Convert ```markdown blocks to <pre> with bold support
    markdown_content = convert_markdown_codeblocks(markdown_content)

    # Extract math formulas and replace with placeholders
    markdown_content, math_formulas = protect_math_formulas(markdown_content)

    # Escape backticks and backslashes for JS template literal
    markdown_content = markdown_content.replace('\\', '\\\\')
    markdown_content = markdown_content.replace('`', '\\`')
    markdown_content = markdown_content.replace('${', '\\${')

    # Convert math formulas to JSON for JavaScript
    import json
    math_formulas_json = json.dumps(math_formulas)

    # Get dates
    now = datetime.now()
    date_iso = now.strftime('%Y-%m-%d')
    date_display = now.strftime('%b %d, %Y')
    year = now.strftime('%Y')

    # Create post directory
    post_dir = os.path.join(repo_root, 'blogs', 'post', folder_name)
    os.makedirs(post_dir, exist_ok=True)

    # Generate HTML
    html = TEMPLATE.format(
        title=title,
        date_iso=date_iso,
        date_display=date_display,
        year=year,
        markdown_content=markdown_content,
        math_formulas_json=math_formulas_json
    )

    # Write HTML file
    html_path = os.path.join(post_dir, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Created: {html_path}")

    # Generate blog entry for blogs/index.html
    entry = BLOG_ENTRY.format(
        folder=folder_name,
        title=title,
        date_iso=date_iso,
        date_display=date_display,
        description=description
    )

    print("\n--- Add this to blogs/index.html (after <div class=\"posts\">) ---\n")
    print(entry)
    print("--- Done ---")

if __name__ == '__main__':
    main()
