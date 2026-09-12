"""KnowLedger 领域异常。"""


class KnowLedgerError(Exception):
    """可预期的领域错误。"""


class ConfigurationError(KnowLedgerError):
    """配置无效或缺失。"""


class ParseError(KnowLedgerError):
    """文档解析失败。"""


class UnsupportedDocumentError(ParseError):
    """当前解析器不支持该文件类型。"""


class DocumentNotFoundError(KnowLedgerError):
    """文档不存在。"""

