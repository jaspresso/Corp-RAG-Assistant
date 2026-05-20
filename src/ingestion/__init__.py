"""
``src.ingestion`` 패키지.

인제션 관련 모듈(``pdf_parser``, ``web_crawler``, ``pipeline``)을 묶는다.
패키지 임포트 시점에 로깅 기본 레벨을 INFO 로 한 번 설정해,
하위 모듈의 ``logging`` 출력이 콘솔에 보이도록 한다.
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
