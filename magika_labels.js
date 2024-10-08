const labels = [
  {
    name: "ai",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "apk",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "appleplist",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "asm",
    threshold: 0.85,
    is_text: true,
  },
  {
    name: "asp",
    threshold: 0.5,
    is_text: true,
  },
  {
    name: "batch",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "bmp",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "bzip",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "c",
    threshold: 0.7,
    is_text: true,
  },
  {
    name: "cab",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "cat",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "chm",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "coff",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "crx",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "cs",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "css",
    threshold: 0.5,
    is_text: true,
  },
  {
    name: "csv",
    threshold: 0.85,
    is_text: true,
  },
  {
    name: "deb",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "dex",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "dmg",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "doc",
    threshold: 0.5,
    is_text: false,
  },
  {
    name: "docx",
    threshold: 0.91,
    is_text: false,
  },
  {
    name: "elf",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "emf",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "eml",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "epub",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "flac",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "gif",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "go",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "gzip",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "hlp",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "html",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "ico",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "ini",
    threshold: 0.85,
    is_text: true,
  },
  {
    name: "internetshortcut",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "iso",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "jar",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "java",
    threshold: 0.91,
    is_text: true,
  },
  {
    name: "javabytecode",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "javascript",
    threshold: 0.8,
    is_text: true,
  },
  {
    name: "jpeg",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "json",
    threshold: 0.75,
    is_text: true,
  },
  {
    name: "latex",
    threshold: 0.5,
    is_text: true,
  },
  {
    name: "lisp",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "lnk",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "m3u",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "macho",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "makefile",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "markdown",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "mht",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "mp3",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "mp4",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "mscompress",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "msi",
    threshold: 0.7,
    is_text: false,
  },
  {
    name: "mum",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "odex",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "odp",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "ods",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "odt",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "ogg",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "outlook",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "pcap",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "pdf",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "pebin",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "pem",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "perl",
    threshold: 0.85,
    is_text: true,
  },
  {
    name: "php",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "png",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "postscript",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "powershell",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "ppt",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "pptx",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "python",
    threshold: 0.85,
    is_text: true,
  },
  {
    name: "pythonbytecode",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "rar",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "rdf",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "rpm",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "rst",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "rtf",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "ruby",
    threshold: 0.93,
    is_text: true,
  },
  {
    name: "rust",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "scala",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "sevenzip",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "shell",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "smali",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "sql",
    threshold: 0.7,
    is_text: true,
  },
  {
    name: "squashfs",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "svg",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "swf",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "symlinktext",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "tar",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "tga",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "tiff",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "torrent",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "ttf",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "txt",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "unknown",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "vba",
    threshold: 0.5,
    is_text: true,
  },
  {
    name: "wav",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "webm",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "webp",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "winregistry",
    threshold: 0.95,
    is_text: true,
  },
  {
    name: "wmf",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "xar",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "xls",
    threshold: 0.8,
    is_text: false,
  },
  {
    name: "xlsb",
    threshold: 0.55,
    is_text: false,
  },
  {
    name: "xlsx",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "xml",
    threshold: 0.9,
    is_text: true,
  },
  {
    name: "xpi",
    threshold: 0.93,
    is_text: false,
  },
  {
    name: "xz",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "yaml",
    threshold: 0.75,
    is_text: true,
  },
  {
    name: "zip",
    threshold: 0.95,
    is_text: false,
  },
  {
    name: "zlibstream",
    threshold: 0.95,
    is_text: false,
  },
];
