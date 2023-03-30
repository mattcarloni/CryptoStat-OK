function userLogOut() {
  localStorage.clear();
}

let cryptoList = [];
let url =
  "https://api.coingecko.com/api/v3/coins/markets?vs_currency=Eur&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=1m";
$.ajax({
  url,
  method: "GET",
  dataType: "json",
  success: function (data) {
    let optionsAdd = [];
    let optionsSell = [];
    data.forEach((item) => {
      cryptoList.push(item);
      let optionAdd = document.createElement("option");
      let optionSell = document.createElement("option");
      optionAdd.setAttribute("value", JSON.stringify([item.name, item.image]));
      optionSell.setAttribute("value", JSON.stringify([item.name, item.image]));
      optionAdd.innerHTML = item.name;
      optionSell.innerHTML = item.name;
      optionsAdd.push(optionAdd);
      optionsSell.push(optionSell);
    });
    document.getElementById("name").replaceChildren(...optionsAdd);
    document.getElementById("nameSell").replaceChildren(...optionsSell);
    document
      .getElementById("cimg")
      .setAttribute("src", JSON.parse(optionsAdd[0].getAttribute("value"))[1]);
    document
      .getElementById("cimgSell")
      .setAttribute("src", JSON.parse(optionsSell[0].getAttribute("value"))[1]);
    $("select").selectize({
      sortField: "text",
    });
  },
});

$("#name").on("change", () => {
  try {
    document
      .getElementById("cimg")
      .setAttribute("src", JSON.parse($("#name").val())[1]);
    document
      .getElementById("cimgSell")
      .setAttribute("src", JSON.parse($("#nameSell").val())[1]);
  } catch {}
});
if (localStorage.getItem("name")) {
  getItems();
} else {
  console.log("index.html");
}

function getItems() {
  (async function () {
    let response = await fetch(`/get-data`);
    let records = await response.json();
    if (response.status === 200) {
      $("#nElementi").html(`${records.length} CRYPTO`);
      let items = [];
      records.forEach((doc) => {
        items.push({
          id: doc._id,
          name: doc.name,
          quantity: doc.quantity,
          purchase_price: doc.purchase_price,
          dob: doc.dob,
        });
      });
      document.getElementById("removeList").innerHTML = "";
      items.forEach((item) => {
        let div = document.createElement("div");
        div.classList.add("crypto__to__remove");
        div.innerHTML = `<div class="name__remove"><i class="fa-solid fa-trash" onclick="deleteCrypto('${item.id}')" style="cursor:pointer"></i> ${item.name} </div><div class="info"><span><h5>Qt: </h5> ${item.quantity} </span><span><h5>Data: </h5> ${item.dob}</span></div>`;
        document.getElementById("removeList").append(div);
      });
    } else console.log("something went wrong");
  })();
}

function addCrypto(c) {
  event.preventDefault();
  try {
    let dob = document.getElementById("dob");
    let name = document.getElementById("name");
    let quantity = document.getElementById("quantity");
    let purchase_price = document.getElementById("purchase_price");
    let proceed = confirm("Are you sure ?");
    if (proceed) {
      let newItem = {
        name: JSON.parse(name.value)[0],
        quantity: quantity.value,
        purchase_price: purchase_price.value,
        dob: dob.value,
      };
      (async function () {
        let response = await fetch("/add-data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(newItem),
        });
        let data = await response.json();
        data.success ? getItems() : console.log("something went wrong");
      })();
    }
  } catch {
    alert("Crypto not found, select one from the list");
  }
}

function addCryptoSold(c) {
  event.preventDefault();
  try {
    let dob = document.getElementById("dos");
    let name = document.getElementById("nameSell");
    let quantity = document.getElementById("quantityS");
    let purchase_price = document.getElementById("purchase_priceS");
    let proceed = confirm("Are you sure ?");
    if (proceed) {
      let newItem = {
        name: JSON.parse(name.value)[0],
        quantity: quantity.value * -1,
        purchase_price: purchase_price.value,
        dob: dob.value,
      };
      (async function () {
        let response = await fetch("/add-data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(newItem),
        });
        let data = await response.json();
        data.success ? getItems() : console.log("something went wrong");
      })();
    }
  } catch {
    alert("Crypto not found, select one from the list");
  }
}


function deleteCrypto(c) {
  let proceed = confirm("Are you sure ?");
  if (proceed) {
    deleteWith();
  }
  async function deleteWith() {
    let response = await fetch(`/delete-data?id=${c}`, {
      method: "DELETE",
    });
    let data = await response.json();
    data.success ? getItems() : console.log("something went wrong");
  }
}

function price() {
  let coin = document.getElementById("name");
  let qnt = document.getElementById("quantity");
  let date = dob.value;

  date = date.split("-").reverse().join("-");
  let coin_name = JSON.parse(coin.value)[0].toLowerCase();

  let url =
    "https://api.coingecko.com/api/v3/coins/" + coin_name + "/history?date=" + date;

  $.ajax({
    url,
    method: "GET",
    dataType: "json",
    success: function (data) {
      if (data.market_data) {
        if (data.market_data.current_price) {
          if (data.market_data.current_price.eur) {
            let singlePrice = data.market_data.current_price.eur;
            let price = singlePrice * qnt.value;
            document.getElementById("purchase_price").value = price;
          }
        }
      }
    },
    error: function (data) {
      console.log(data);
    },
  });
}

function priceSell() {
  let coin = document.getElementById("nameSell");
  let qnt = document.getElementById("quantityS");
  let date = dos.value;

  date = date.split("-").reverse().join("-");
  let coin_name = JSON.parse(coin.value)[0].toLowerCase();

  let url =
    "https://api.coingecko.com/api/v3/coins/" + coin_name + "/history?date=" + date;

  $.ajax({
    url,
    method: "GET",
    dataType: "json",
    success: function (data) {
      if (data.market_data) {
        if (data.market_data.current_price) {
          if (data.market_data.current_price.eur) {
            let singlePrice = data.market_data.current_price.eur;
            let price = singlePrice * qnt.value * -1;
            document.getElementById("purchase_priceS").value = price;
          }
        }
      }
    },
    error: function (data) {
      console.log(data);
    },
  });
}
