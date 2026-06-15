from datetime import datetime
from typing import List, Tuple

from skmap.misc import date_range


class TestDateRange:
    def _pluck_year(
        self, dates: List[Tuple[datetime, datetime]], sep: str = "-"
    ) -> str:
        result = []

        for dt1, dt2 in dates:
            if isinstance(dt1, datetime):
                result.append(f"{dt1.month}{dt1.day}{dt2.month}{dt2.day}")
            else:
                result.append(f"{dt1[4:]}{dt2[4:]}")

        return sep.join(result)

    def test_basic(self) -> None:
        assert self._pluck_year(
            date_range("2013-01-01", "2016-01-01", "months", 1, ignore_29feb=True)
        ) == self._pluck_year(
            date_range("2016-01-01", "2019-01-01", "months", 1, ignore_29feb=True)
        )

    def test_leap_year(self) -> None:
        assert self._pluck_year(
            date_range("2013-01-01", "2016-01-01", "months", 1, ignore_29feb=True)
        ) != self._pluck_year(
            date_range("2016-01-01", "2019-01-01", "months", 1, ignore_29feb=False)
        )

    def test_Yj_format(self) -> None:
        assert self._pluck_year(
            date_range(
                "2013001", "2016001", "months", 1, date_format="%Y%j", ignore_29feb=True
            )
        ) == self._pluck_year(
            date_range(
                "2016001", "2019001", "months", 1, date_format="%Y%j", ignore_29feb=True
            )
        )

    def test_Yj_leap_year(self) -> None:
        assert self._pluck_year(
            date_range(
                "2013001", "2016001", "months", 1, date_format="%Y%j", ignore_29feb=True
            )
        ) != self._pluck_year(
            date_range(
                "2016001",
                "2019001",
                "months",
                1,
                date_format="%Y%j",
                ignore_29feb=False,
            )
        )

    def test_date_step(self) -> None:
        date_step = ([16] * 22) + [13]
        assert self._pluck_year(
            date_range(
                "2013001",
                "2016001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=True,
            )
        ) == self._pluck_year(
            date_range(
                "2016001",
                "2019001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=True,
            )
        )

    def test_date_step_leap_year(self) -> None:
        date_step = ([16] * 22) + [13]
        assert self._pluck_year(
            date_range(
                "2013001",
                "2016001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=True,
            )
        ) != self._pluck_year(
            date_range(
                "2016001",
                "2019001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=False,
            )
        )

    def test_date_step_str(self) -> None:
        date_step = ([16] * 22) + [13]
        assert self._pluck_year(
            date_range(
                "2013001",
                "2016001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=True,
                return_str=True,
            )
        ) == self._pluck_year(
            date_range(
                "2016001",
                "2019001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=True,
                return_str=True,
            )
        )

    def test_date_step_str_leap_year(self) -> None:
        date_step = ([16] * 22) + [13]
        assert self._pluck_year(
            date_range(
                "2013001",
                "2016001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=True,
                return_str=True,
            )
        ) != self._pluck_year(
            date_range(
                "2016001",
                "2019001",
                "days",
                date_step,
                date_format="%Y%j",
                ignore_29feb=False,
                return_str=True,
            )
        )
