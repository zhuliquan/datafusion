// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Regx expressions
use arrow::array::{Array, ArrayDataBuilder, ArrayRef, BooleanArray, GenericStringArray, OffsetSizeTrait};
use arrow::compute::kernels::regexp;
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use arrow_buffer::{BooleanBufferBuilder, NullBuffer};
use datafusion_common::exec_err;
use datafusion_common::ScalarValue;
use datafusion_common::{arrow_datafusion_err, plan_err};
use datafusion_common::{
    cast::as_generic_string_array, internal_err, DataFusionError, Result,
};
use datafusion_expr::ColumnarValue;
use datafusion_expr::TypeSignature::*;
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};
use regex;
use lru::LruCache;
use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[derive(Debug)]
pub struct RegexpLikeFunc {
    signature: Signature,
    patterns: LruCache<String, regex::Regex>,
}
impl Default for RegexpLikeFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl RegexpLikeFunc {
    pub fn new() -> Self {
        use DataType::*;
        Self {
            signature: Signature::one_of(
                vec![
                    Exact(vec![Utf8, Utf8]),
                    Exact(vec![LargeUtf8, Utf8]),
                    Exact(vec![Utf8, Utf8, Utf8]),
                    Exact(vec![LargeUtf8, Utf8, Utf8]),
                ],
                Volatility::Immutable,
            ),
            // TODO: may be make cache_size as constant
            patterns: LruCache::new(NonZeroUsize::new(100 as usize).unwrap()),
        }
    }
}

impl ScalarUDFImpl for RegexpLikeFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "regexp_like"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        use DataType::*;

        Ok(match &arg_types[0] {
            LargeUtf8 | Utf8 => Boolean,
            Null => Null,
            other => {
                return plan_err!(
                    "The regexp_like function can only accept strings. Got {other}"
                );
            }
        })
    }
    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let len = args
            .iter()
            .fold(Option::<usize>::None, |acc, arg| match arg {
                ColumnarValue::Scalar(_) => acc,
                ColumnarValue::Array(a) => Some(a.len()),
            });

        let is_scalar = len.is_none();
        let inferred_length = len.unwrap_or(1);
        let args = args
            .iter()
            .map(|arg| arg.clone().into_array(inferred_length))
            .collect::<Result<Vec<_>>>()?;

        let result = self.regexp_like_func(&args);
        if is_scalar {
            // If all inputs are scalar, keeps output as scalar
            let result = result.and_then(|arr| ScalarValue::try_from_array(&arr, 0));
            result.map(ColumnarValue::Scalar)
        } else {
            result.map(ColumnarValue::Array)
        }
    }
}

impl RegexpLikeFunc {
    fn regexp_like_func(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        match args[0].data_type() {
            DataType::Utf8 => self.regexp_like::<i32>(args),
            DataType::LargeUtf8 => self.regexp_like::<i64>(args),
            other => {
                internal_err!("Unsupported data type {other:?} for function regexp_like")
            }
        }
    }

    /// Tests a string using a regular expression returning true if at
    /// least one match, false otherwise.
    ///
    /// The full list of supported features and syntax can be found at
    /// <https://docs.rs/regex/latest/regex/#syntax>
    ///
    /// Supported flags can be found at
    /// <https://docs.rs/regex/latest/regex/#grouping-and-flags>
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use datafusion::prelude::*;
    /// # use datafusion::error::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// let ctx = SessionContext::new();
    /// let df = ctx.read_csv("tests/data/regex.csv", CsvReadOptions::new()).await?;
    ///
    /// // use the regexp_like function to test col 'values',
    /// // against patterns in col 'patterns' without flags
    /// let df = df.with_column(
    ///     "a",
    ///     regexp_like(vec![col("values"), col("patterns")])
    /// )?;
    /// // use the regexp_like function to test col 'values',
    /// // against patterns in col 'patterns' with flags
    /// let df = df.with_column(
    ///     "b",
    ///     regexp_like(vec![col("values"), col("patterns"), col("flags")])
    /// )?;
    /// // literals can be used as well with dataframe calls
    /// let df = df.with_column(
    ///     "c",
    ///     regexp_like(vec![lit("foobarbequebaz"), lit("(bar)(beque)")])
    /// )?;
    ///
    /// df.show().await?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn regexp_like<T: OffsetSizeTrait>(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        match args.len() {
            2 => {
                let values = as_generic_string_array::<T>(&args[0])?;
                let regex = as_generic_string_array::<T>(&args[1])?;
                let array = regexp::regexp_is_match_utf8(values, regex, None)
                    .map_err(|e| arrow_datafusion_err!(e))?;
    
                Ok(Arc::new(array) as ArrayRef)
            }
            3 => {
                let values = as_generic_string_array::<T>(&args[0])?;
                let regex = as_generic_string_array::<T>(&args[1])?;
                let flags = as_generic_string_array::<T>(&args[2])?;
    
                if flags.iter().any(|s| s == Some("g")) {
                    return plan_err!("regexp_like() does not support the \"global\" option");
                }
    
                let array = regexp::regexp_is_match_utf8(values, regex, Some(flags))
                    .map_err(|e| arrow_datafusion_err!(e))?;
    
                Ok(Arc::new(array) as ArrayRef)
            }
            other => exec_err!(
                "regexp_like was called with {other} arguments. It requires at least 2 and at most 3."
            ),
        }
    }

    /// regexp_is_match_utf8 copyed from arrow-rs
    fn regexp_is_match_utf8<OffsetSize: OffsetSizeTrait>(
        &self,
        array: &GenericStringArray<OffsetSize>,
        regex_array: &GenericStringArray<OffsetSize>,
        flags_array: Option<&GenericStringArray<OffsetSize>>,
    ) -> Result<BooleanArray, ArrowError> {
        if array.len() != regex_array.len() {
            return Err(ArrowError::ComputeError(
                "Cannot perform comparison operation on arrays of different length".to_string(),
            ));
        }
        let nulls = NullBuffer::union(array.nulls(), regex_array.nulls());
    
        let mut result = BooleanBufferBuilder::new(array.len());
    
        let complete_pattern = match flags_array {
            Some(flags) => Box::new(
                regex_array
                    .iter()
                    .zip(flags.iter())
                    .map(|(pattern, flags)| {
                        pattern.map(|pattern| match flags {
                            Some(flag) => format!("(?{flag}){pattern}"),
                            None => pattern.to_string(),
                        })
                    }),
            ) as Box<dyn Iterator<Item = Option<String>>>,
            None => Box::new(
                regex_array
                    .iter()
                    .map(|pattern| pattern.map(|pattern| pattern.to_string())),
            ),
        };
    
        array
            .iter()
            .zip(complete_pattern)
            .map(|(value, pattern)| {
                match (value, pattern) {
                    // Required for Postgres compatibility:
                    // SELECT 'foobarbequebaz' ~ ''); = true
                    (Some(_), Some(pattern)) if pattern == *"" => {
                        result.append(true);
                    }
                    (Some(value), Some(pattern)) => {
                        let existing_pattern = self.patterns.get(&pattern);
                        let re = match existing_pattern {
                            Some(re) => re,
                            None => {
                                let re = regex::Regex::new(pattern.as_str()).map_err(|e| {
                                    ArrowError::ComputeError(format!(
                                        "Regular expression did not compile: {e:?}"
                                    ))
                                })?;
                                self.patterns.put(pattern, re.clone());
                                &re
                            }
                        };
                        result.append(re.is_match(value));
                    }
                    _ => result.append(false),
                }
                Ok(())
            })
            .collect::<Result<Vec<()>, ArrowError>>()?;
    
        let data = unsafe {
            ArrayDataBuilder::new(DataType::Boolean)
                .len(array.len())
                .buffers(vec![result.into()])
                .nulls(nulls)
                .build_unchecked()
        };
        Ok(BooleanArray::from(data))
    }
}


// #[cfg(test)]
// mod tests {
//     use std::sync::Arc;

//     use arrow::array::BooleanBuilder;
//     use arrow::array::StringArray;


//     #[test]
//     fn test_case_sensitive_regexp_like() {
//         let values = StringArray::from(vec!["abc"; 5]);

//         let patterns =
//             StringArray::from(vec!["^(a)", "^(A)", "(b|d)", "(B|D)", "^(b|c)"]);

//         let mut expected_builder: BooleanBuilder = BooleanBuilder::new();
//         expected_builder.append_value(true);
//         expected_builder.append_value(false);
//         expected_builder.append_value(true);
//         expected_builder.append_value(false);
//         expected_builder.append_value(false);
//         let expected = expected_builder.finish();

//         let re = regexp_like::<i32>(&[Arc::new(values), Arc::new(patterns)]).unwrap();

//         assert_eq!(re.as_ref(), &expected);
//     }

//     #[test]
//     fn test_case_insensitive_regexp_like() {
//         let values = StringArray::from(vec!["abc"; 5]);
//         let patterns =
//             StringArray::from(vec!["^(a)", "^(A)", "(b|d)", "(B|D)", "^(b|c)"]);
//         let flags = StringArray::from(vec!["i"; 5]);

//         let mut expected_builder: BooleanBuilder = BooleanBuilder::new();
//         expected_builder.append_value(true);
//         expected_builder.append_value(true);
//         expected_builder.append_value(true);
//         expected_builder.append_value(true);
//         expected_builder.append_value(false);
//         let expected = expected_builder.finish();

//         let re =
//             regexp_like::<i32>(&[Arc::new(values), Arc::new(patterns), Arc::new(flags)])
//                 .unwrap();

//         assert_eq!(re.as_ref(), &expected);
//     }

//     #[test]
//     fn test_unsupported_global_flag_regexp_like() {
//         let values = StringArray::from(vec!["abc"]);
//         let patterns = StringArray::from(vec!["^(a)"]);
//         let flags = StringArray::from(vec!["g"]);

//         let re_err =
//             regexp_like::<i32>(&[Arc::new(values), Arc::new(patterns), Arc::new(flags)])
//                 .expect_err("unsupported flag should have failed");

//         assert_eq!(
//             re_err.strip_backtrace(),
//             "Error during planning: regexp_like() does not support the \"global\" option"
//         );
//     }
// }
