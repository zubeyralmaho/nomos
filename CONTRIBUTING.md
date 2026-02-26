# Contributing to Nomos

Nomos'a katkıda bulunmak istediğiniz için teşekkürler! Bu rehber size yardımcı olacak.

## Geliştirme Ortamı

### Gereksinimler

- Rust 1.82+
- Cargo
- Git

### Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/zubeyralmaho/nomos.git
cd nomos

# Bağımlılıkları yükle
cargo build

# Testleri çalıştır
cargo test
```

## Proje Yapısı

```
nomos/
├── nomos-core/           # Ana kütüphane
│   └── src/
│       ├── nlp/          # NLP algoritmaları
│       │   ├── levenshtein.rs
│       │   ├── jaro_winkler.rs
│       │   ├── ngram.rs
│       │   ├── soundex.rs
│       │   └── metaphone.rs
│       ├── engine/       # Healing engine
│       │   ├── matcher.rs
│       │   └── confidence.rs
│       ├── proxy.rs      # HTTP proxy
│       └── middleware.rs # Healing middleware
├── dashboard/            # Web UI
├── docs/                 # Dokümantasyon
└── bench.sh             # Benchmark scripti
```

## Kod Standartları

### Rust

- `cargo fmt` ile formatla
- `cargo clippy` ile lint kontrolü yap
- Her fonksiyona dokümantasyon ekle
- Test yaz (en az %80 coverage)

```rust
/// Field isimlerini karşılaştırır.
///
/// # Arguments
/// * `source` - Kaynak field ismi
/// * `target` - Hedef field ismi
///
/// # Returns
/// Benzerlik skoru (0.0-1.0)
pub fn compare(source: &str, target: &str) -> f64 {
    // ...
}
```

### Commit Mesajları

Format: `<tip>: <açıklama>`

Tipler:
- `feat`: Yeni özellik
- `fix`: Bug düzeltme
- `docs`: Dokümantasyon
- `test`: Test ekleme/güncelleme
- `refactor`: Kod düzenleme
- `perf`: Performans iyileştirme
- `chore`: Genel bakım

Örnekler:
```
feat: add metaphone algorithm
fix: handle empty strings in jaro-winkler
docs: update API documentation
test: add edge cases for ngram
perf: optimize levenshtein with SIMD
```

## Pull Request Süreci

1. **Fork** yapın
2. Feature branch oluşturun: `git checkout -b feat/yeni-ozellik`
3. Değişiklikleri yapın
4. Testleri çalıştırın: `cargo test`
5. Linting: `cargo clippy`
6. Commit yapın
7. Push: `git push origin feat/yeni-ozellik`
8. Pull Request açın

### PR Checklist

- [ ] Testler geçiyor
- [ ] Dokümantasyon güncellendi
- [ ] `cargo fmt` çalıştırıldı
- [ ] `cargo clippy` uyarısı yok
- [ ] Benchmark regresyon yok

## Yeni NLP Algoritması Ekleme

1. `nomos-core/src/nlp/` altında yeni dosya oluştur
2. `SimilarityAlgorithm` trait'ini implemente et
3. `mod.rs`'de export et
4. Testler ekle
5. `engine/confidence.rs`'de ensemble'a ekle

```rust
// nomos-core/src/nlp/yeni_algo.rs

pub struct YeniAlgo;

impl YeniAlgo {
    pub fn similarity(s1: &str, s2: &str) -> f64 {
        // Algoritma implementasyonu
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        assert_eq!(YeniAlgo::similarity("test", "test"), 1.0);
    }

    #[test]
    fn test_different() {
        assert!(YeniAlgo::similarity("abc", "xyz") < 0.5);
    }
}
```

## Test Yazma

### Unit Testler

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let result = some_function("input");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_edge_case() {
        assert_eq!(some_function(""), expected_for_empty);
    }
}
```

### Benchmark Testler

```bash
# Benchmark çalıştır
./bench.sh

# Belirli test
cargo bench --bench proxy_throughput
```

## Issue Açma

Bug raporu için:
- Rust versiyonu
- OS bilgisi
- Adımlar
- Beklenen davranış
- Gerçekleşen davranış

Feature isteği için:
- Ne istediğinizi açıklayın
- Neden gerekli olduğunu belirtin
- Varsa örnek kullanım senaryosu

## Sorular

Sorularınız için:
- GitHub Discussions kullanın
- Issue açmadan önce mevcut issue'ları kontrol edin

## Lisans

Katkılarınız MIT lisansı altında olacaktır.

---

Katkılarınız için teşekkürler! 🚀
